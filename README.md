# Watermark-Forgery-Attack
# Reproducing the Submission

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Get the dataset

Download and unzip the task dataset so that the following structure exists:

```
Dataset/
  clean_targets/
  watermarked_sources/
    WM_1/ ... WM_8/
```

## 3. Run the pipeline

```bash
python forge_watermarks.py --dataset_dir /path/to/Dataset --out_dir ./submission_out --zip_path ./submission.zip
```

This produces:
- `./submission_out/` : the 200 forged PNGs
- `./submission.zip` : the ready to submit zip file
- `./submission_out/run_report.json` : per group method/reliability log

## 4. Submit 

Edit `YOUR_API_KEY_HERE` in `submit_zip()` inside `forge_watermarks.py`, then call:

```python
submit_zip("./submission.zip")


Reuirements:
numpy
pillow
opencv-python
torch
lpips
PyWavelets
invisible-watermark
onnxruntime
trustmark
