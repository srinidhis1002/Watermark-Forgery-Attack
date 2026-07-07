import argparse
import itertools
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pywt
import torch
import lpips as lpips_lib
from PIL import Image
from imwatermark import WatermarkEncoder, WatermarkDecoder
from trustmark import TrustMark

# CONFIG

CATEGORIES = [
    ("WM_1", 1, 25), ("WM_2", 26, 50), ("WM_3", 51, 75), ("WM_4", 76, 100),
    ("WM_5", 101, 125), ("WM_6", 126, 150), ("WM_7", 151, 175), ("WM_8", 176, 200),
]

LPIPS_BUDGET = 0.06
ALPHA_START = 0.5
ALPHA_MAX_DOUBLINGS = 25

KNOWN_METHODS = ["dwtDct", "dwtDctSvd", "rivaGan"]
KNOWN_LENGTHS = [8, 16, 24, 32, 40, 48, 64, 72, 80, 96, 128, 160, 192, 256]
CONFIRM_THRESHOLD = 0.75
MIN_DECODE_SIZE = 256

# Confirmed via fingerprinting on this dataset; re-verified live in main(),
# not trusted blindly.
KNOWN_OVERRIDES = {
    "WM_1": {"method": "dwtDct", "length": 16},
    "WM_2": {"method": "rivaGan", "length": 32},
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
# IO HELPERS
 
def load_rgb(path):
    return np.array(Image.open(path).convert("RGB")).astype(np.float32)

def load_bgr(path):
    return cv2.imread(str(path))

 
# STATISTICAL EXTRACTION, denoising-residual method bank
 
def trimmed_mean(stack, trim_frac=0.1):
    n = stack.shape[0]
    k = int(np.floor(n * trim_frac))
    s = np.sort(stack, axis=0)
    return (s[k:n - k] if k > 0 else s).mean(axis=0)

def normalized_correlation(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))

def texture_mask(img_rgb):
    gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.sqrt(gx ** 2 + gy ** 2), (9, 9), 0)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return 0.4 + 0.6 * mag

def self_referencing_residual(img_rgb):
    u = np.clip(img_rgb, 0, 255).astype(np.uint8)
    d = cv2.fastNlMeansDenoisingColored(u, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)
    return img_rgb - d.astype(np.float32)

def residual_bilateral(img_rgb):
    u = np.clip(img_rgb, 0, 255).astype(np.uint8)
    return img_rgb - cv2.bilateralFilter(u, 9, 15, 15).astype(np.float32)

def residual_median(img_rgb, ksize=3):
    u = np.clip(img_rgb, 0, 255).astype(np.uint8)
    return img_rgb - cv2.medianBlur(u, ksize).astype(np.float32)

def residual_wavelet(img_rgb, wavelet='db4', level=2):
    out = np.zeros_like(img_rgb)
    for c in range(3):
        coeffs = pywt.wavedec2(img_rgb[..., c], wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745
        thresh = sigma * np.sqrt(2 * np.log(img_rgb[..., c].size))
        nc = [coeffs[0]] + [tuple(pywt.threshold(d, thresh, mode='soft') for d in det) for det in coeffs[1:]]
        den = pywt.waverec2(nc, wavelet)[:img_rgb.shape[0], :img_rgb.shape[1]]
        out[..., c] = img_rgb[..., c] - den
    return out

def residual_dct_highfreq(img_rgb, keep_frac=0.35):
    out = np.zeros_like(img_rgb)
    for c in range(3):
        d = cv2.dct(img_rgb[..., c])
        h, w = d.shape
        m = np.ones_like(d)
        m[:int(h * keep_frac), :int(w * keep_frac)] = 0
        out[..., c] = cv2.idct(d * m)
    return out

def residual_luma_nlm(img_rgb):
    u = np.clip(img_rgb, 0, 255).astype(np.uint8)
    ycc = cv2.cvtColor(u, cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0]
    y_den = cv2.fastNlMeansDenoising(y, None, h=6, templateWindowSize=7, searchWindowSize=21)
    out = np.zeros_like(img_rgb)
    out[..., 0] = y.astype(np.float32) - y_den.astype(np.float32)
    return out

EXTRACTION_METHODS = {
    "nlm": self_referencing_residual,
    "bilateral": residual_bilateral,
    "median3": lambda i: residual_median(i, 3),
    "median5": lambda i: residual_median(i, 5),
    "wavelet": residual_wavelet,
    "dct_hf": residual_dct_highfreq,
    "luma_nlm": residual_luma_nlm,
}

def extract_with_method(stack, fn):
    rs = np.stack([fn(stack[i]) for i in range(stack.shape[0])], axis=0)
    return 0.5 * trimmed_mean(rs, 0.1) + 0.5 * np.median(rs, axis=0)

def split_half_reliability_method(stack, fn, n_trials=3, seed=0):
    rng = np.random.default_rng(seed)
    n = stack.shape[0]
    corrs = []
    for _ in range(n_trials):
        idx = rng.permutation(n)
        half = n // 2
        ea = extract_with_method(stack[idx[:half]], fn)
        eb = extract_with_method(stack[idx[half:]], fn)
        corrs.append(normalized_correlation(ea, eb))
    return float(np.mean(corrs))


# Periodicity detection) + tile-averaging
def profile_autocorr_peak(profile, min_lag=4, max_lag=None):
    n = len(profile)
    if max_lag is None:
        max_lag = n // 2
    x = profile - profile.mean()
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 1e-8:
        return None, 0.0
    ac_norm = ac / ac[0]
    search = ac_norm[min_lag:max_lag]
    if len(search) == 0:
        return None, 0.0
    best_lag = int(np.argmax(search)) + min_lag
    return best_lag, float(search[best_lag - min_lag])

def _peak_score(img, min_lag=4):
    h, w, _ = img.shape
    _, sh = profile_autocorr_peak(img.mean(axis=(1, 2)), min_lag, h // 2)
    _, sw = profile_autocorr_peak(img.mean(axis=(0, 2)), min_lag, w // 2)
    return min(sh, sw)

def detect_periodicity(mean_residual, min_lag=4, n_null=20, z_threshold=3.0, seed=0):
    rng = np.random.default_rng(seed)
    h, w, _ = mean_residual.shape
    real_score = _peak_score(mean_residual, min_lag)
    null_scores = [
        _peak_score(np.roll(mean_residual,
                             (rng.integers(min_lag, h - min_lag), rng.integers(min_lag, w - min_lag)),
                             axis=(0, 1)), min_lag)
        for _ in range(n_null)
    ]
    z = (real_score - np.mean(null_scores)) / (np.std(null_scores) + 1e-8)
    if z < z_threshold:
        return None
    lag_h, _ = profile_autocorr_peak(mean_residual.mean(axis=(1, 2)), min_lag, h // 2)
    lag_w, _ = profile_autocorr_peak(mean_residual.mean(axis=(0, 2)), min_lag, w // 2)
    return {"period_h": lag_h, "period_w": lag_w, "score": real_score, "z": z}

def tile_average_extract(stack, ph, pw, sigma=3):
    n, h, w, c = stack.shape
    ch_, cw_ = (h // ph) * ph, (w // pw) * pw
    tsum, tcount = np.zeros((ph, pw, c)), 0
    for i in range(n):
        img = stack[i, :ch_, :cw_, :]
        hp = img - cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
        r = hp.reshape(ch_ // ph, ph, cw_ // pw, pw, c)
        tsum += r.sum(axis=(0, 2))
        tcount += r.shape[0] * r.shape[2]
    tile = (tsum / tcount).astype(np.float32)
    return np.tile(tile, (int(np.ceil(h / ph)), int(np.ceil(w / pw)), 1))[:h, :w, :]

def split_half_reliability_tile(stack, ph, pw, n_trials=3, seed=0):
    rng = np.random.default_rng(seed)
    n = stack.shape[0]
    corrs = []
    for _ in range(n_trials):
        idx = rng.permutation(n)
        half = n // 2
        ea = tile_average_extract(stack[idx[:half]], ph, pw)
        eb = tile_average_extract(stack[idx[half:]], ph, pw)
        corrs.append(normalized_correlation(ea, eb))
    return float(np.mean(corrs))

def extract_best_statistical_ensemble(stack, verbose=True):

    candidates = {}
    for name, fn in EXTRACTION_METHODS.items():
        rel = split_half_reliability_method(stack, fn)
        candidates[name] = rel
        if verbose:
            print(f"    {name:10s} reliability={rel:.3f}")

    best_name = max(candidates, key=candidates.get)
    probe = extract_with_method(stack, EXTRACTION_METHODS[best_name])
    period_info = detect_periodicity(probe)

    if period_info is not None:
        ph, pw = period_info["period_h"], period_info["period_w"]
        rel_tile = split_half_reliability_tile(stack, ph, pw)
        candidates["tile_avg"] = rel_tile
        if verbose:
            print(f"    periodicity SIGNIFICANT: period=({ph},{pw}) z={period_info['z']:.2f} "
                  f"-> tile_avg={rel_tile:.3f}")
    else:
        period_info = None
        if verbose:
            print("    no significant periodicity")

    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    top2 = ranked[:2]
    if verbose:
        print(f"    ensembling top-2: {top2}")

    def get_residual(name):
        if name == "tile_avg":
            return tile_average_extract(stack, period_info["period_h"], period_info["period_w"])
        return extract_with_method(stack, EXTRACTION_METHODS[name])

    w1, w2 = top2[0][1] ** 2, top2[1][1] ** 2
    total_w = w1 + w2 + 1e-8
    residual = (w1 * get_residual(top2[0][0]) + w2 * get_residual(top2[1][0])) / total_w
    best_rel = top2[0][1]
    method_label = f"{top2[0][0]}+{top2[1][0]}"
    return residual, method_label, best_rel


 
# KNOWN-SCHEME FINGERPRINTING (invisible-watermark + TrustMark)
 
def decode_bits_safe(img_bgr, method, length, min_size=MIN_DECODE_SIZE):
    h, w = img_bgr.shape[:2]
    if h < min_size or w < min_size:
        scale = min_size / min(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale) + 1, int(h * scale) + 1),
                              interpolation=cv2.INTER_LANCZOS4)
    try:
        return np.array(WatermarkDecoder('bits', length).decode(img_bgr, method), dtype=np.int8)
    except Exception:
        return None

def group_bit_agreement(paths, method, length):
    decoded = [b for b in (decode_bits_safe(load_bgr(p), method, length) for p in paths) if b is not None]
    if len(decoded) < 2:
        return None
    return float(np.mean([np.mean(a == b) for a, b in itertools.combinations(decoded, 2)]))

def fingerprint_group(src_paths, methods=KNOWN_METHODS, lengths=KNOWN_LENGTHS):
    best = {"method": None, "length": None, "agreement": 0.0}
    for method in methods:
        for length in lengths:
            agree = group_bit_agreement(src_paths, method, length)
            if agree is not None and agree > best["agreement"]:
                best = {"method": method, "length": length, "agreement": agree}
    return best

def recover_message_bits(paths, method, length):
    votes = [decode_bits_safe(load_bgr(p), method, length) for p in paths]
    votes = np.stack([v for v in votes if v is not None], axis=0)
    majority = (votes.sum(axis=0) > votes.shape[0] / 2).astype(np.int8)
    return majority, float(np.mean(votes == majority[None, :]))

def fingerprint_trustmark(tm, src_paths):
    decoded = []
    for p in src_paths:
        try:
            result = tm.decode(Image.open(p).convert("RGB"))
            if result[1]:
                decoded.append(result[0])
        except Exception:
            continue
    if len(decoded) < 2:
        return {"agreement": 0.0, "secret": None}
    agreements = [a == b for a, b in itertools.combinations(decoded, 2)]
    return {"agreement": float(np.mean(agreements)), "secret": decoded[0] if decoded else None}


 
# LPIPS + ADAPTIVE INJECTION
 
_lpips_fn = None

def get_lpips_fn():
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips_lib.LPIPS(net='alex').to(DEVICE).eval()
    return _lpips_fn

def to_lpips_tensor(arr):
    return (torch.from_numpy(arr).float() / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def lpips_distance(a, b):
    fn = get_lpips_fn()
    return float(fn(to_lpips_tensor(a), to_lpips_tensor(b)).item())

def find_best_alpha(target_arr, residual, mask, lpips_budget,
                     start_alpha=ALPHA_START, max_doublings=ALPHA_MAX_DOUBLINGS):
    """Exponential ramp then binary refine -- finds the largest alpha whose
    LPIPS distance to the clean image stays under budget."""
    alpha, prev_alpha = start_alpha, 0.0
    last_good = {"alpha": 0.0, "lpips": 0.0}
    for _ in range(max_doublings):
        forged = np.clip(target_arr + alpha * mask[..., None] * residual, 0, 255)
        lp = lpips_distance(target_arr.astype(np.uint8), forged.astype(np.uint8))
        if lp > lpips_budget:
            break
        last_good = {"alpha": float(alpha), "lpips": float(lp)}
        prev_alpha = alpha
        alpha *= 2
    else:
        return last_good
    lo, hi = prev_alpha, alpha
    for _ in range(10):
        mid = (lo + hi) / 2
        forged = np.clip(target_arr + mid * mask[..., None] * residual, 0, 255)
        lp = lpips_distance(target_arr.astype(np.uint8), forged.astype(np.uint8))
        if lp <= lpips_budget:
            lo = mid
            last_good = {"alpha": float(mid), "lpips": float(lp)}
        else:
            hi = mid
    return last_good


 
# MAIN PIPELINE
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                         help="Path to folder containing clean_targets/ and watermarked_sources/")
    parser.add_argument("--out_dir", type=str, default="./submission_out")
    parser.add_argument("--zip_path", type=str, default="./submission.zip")
    parser.add_argument("--lpips_budget", type=float, default=LPIPS_BUDGET)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    clean_dir = dataset_dir / "clean_targets"
    source_dir = dataset_dir / "watermarked_sources"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    WatermarkEncoder.loadModel()  # required before any rivaGan call works
    tm = TrustMark(verbose=False, model_type='Q')

    report = {}

    for wm_name, start, stop in CATEGORIES:
        print(f"=== {wm_name} (images {start}-{stop}) ===")
        src_paths = sorted((source_dir / wm_name).glob("*.png"))
        assert len(src_paths) == 25, f"{wm_name}: expected 25 source images, found {len(src_paths)}"

        # --- Stage 1: fingerprinting ---
        if wm_name in KNOWN_OVERRIDES:
            cand = KNOWN_OVERRIDES[wm_name]
            agree = group_bit_agreement(src_paths, cand["method"], cand["length"]) or 0.0
            fp = {**cand, "agreement": agree}
        else:
            fp = fingerprint_group(src_paths)

        if fp["agreement"] <= CONFIRM_THRESHOLD:
            tm_result = fingerprint_trustmark(tm, src_paths)
            if tm_result["agreement"] > fp["agreement"]:
                fp = {"method": "trustmark", "length": None, "agreement": tm_result["agreement"],
                      "secret": tm_result["secret"]}

        # --- Stage 2: injection ---
        if fp["agreement"] > CONFIRM_THRESHOLD and fp.get("method") == "trustmark":
            print(f"  REAL ENCODER (TrustMark), agreement={fp['agreement']:.3f}")
            for num in range(start, stop + 1):
                fname = f"{num}.png"
                clean_img = Image.open(clean_dir / fname).convert("RGB")
                forged_img = tm.encode(clean_img, fp["secret"], MODE='text')
                forged_img.save(out_dir / fname)
            report[wm_name] = {"mode": "trustmark", "fingerprint_agreement": fp["agreement"]}

        elif fp["agreement"] > CONFIRM_THRESHOLD:
            method, length = fp["method"], fp["length"]
            message_bits, msg_conf = recover_message_bits(src_paths, method, length)
            print(f"  REAL ENCODER: {method} len={length} "
                  f"(fingerprint_agreement={fp['agreement']:.3f}, message_confidence={msg_conf:.3f})")
            encoder = WatermarkEncoder()
            encoder.set_watermark('bits', message_bits.tolist())
            for num in range(start, stop + 1):
                fname = f"{num}.png"
                target_bgr = cv2.imread(str(clean_dir / fname))
                forged_bgr = encoder.encode(target_bgr, method)
                cv2.imwrite(str(out_dir / fname), forged_bgr)
            report[wm_name] = {"mode": "real_encoder", "method": method, "length": length,
                                "fingerprint_agreement": fp["agreement"], "message_confidence": msg_conf}

        else:
            print("  no known-scheme match -> statistical extraction")
            stack = np.stack([load_rgb(p) for p in src_paths], axis=0)
            residual, method, reliability = extract_best_statistical_ensemble(stack)
            print(f"  -> chose '{method}' reliability={reliability:.3f}")
            for num in range(start, stop + 1):
                fname = f"{num}.png"
                target_arr = load_rgb(clean_dir / fname)
                mask = texture_mask(target_arr)
                result = find_best_alpha(target_arr, residual, mask, args.lpips_budget)
                forged = np.clip(target_arr + result["alpha"] * mask[..., None] * residual, 0, 255).astype(np.uint8)
                Image.fromarray(forged).save(out_dir / fname)
            report[wm_name] = {"mode": "statistical", "method": method, "reliability": reliability}
        print()

    #  Package submission zip 
    png_files = sorted(out_dir.glob("*.png"), key=lambda p: int(p.stem))
    assert len(png_files) == 200, f"Expected 200 forged images, found {len(png_files)}"

    zip_path = Path(args.zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in png_files:
            zf.write(img_path, arcname=img_path.name)

    with open(out_dir / "run_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Done. {len(png_files)} images forged.")
    print(f"Submission zip: {zip_path.resolve()}")

# SUBMISSION CODE

def submit_zip(zip_path, api_key="YOUR_API_KEY_HERE",
               base_url="http://34.63.153.158", task_id="22-forging-task"):
    import requests
    with open(zip_path, "rb") as f:
        resp = requests.post(f"{base_url}/submit/{task_id}", headers={"X-API-Key": api_key},
                              files={"file": (Path(zip_path).name, f, "zip")})
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}
    print("Response:", body)
    return body

if __name__ == "__main__":
    main()
