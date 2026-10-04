# adaptive-dnn-inference

Adaptive Deep Neural Network Inference using Confidence-Based Early Exits (ICT-4442 Deep Learning project).

Images exit a network at an intermediate classifier once the prediction is confident enough, so easy
inputs use less computation. We compare, on CIFAR-100:

1. ResNet-18 with full inference (baseline)
2. Confidence-based early-exit ResNet-18 (max softmax probability threshold)
3. Calibrated / entropy-based early-exit ResNet-18 (per-exit temperature scaling, entropy threshold)
4. Early-exit ViT-Tiny

and measure accuracy, FLOPs, real latency (GPU and CPU, batch 1) and behaviour under input corruptions.

## Layout

```
src/data.py          CIFAR-100 download, stratified 45k/5k train/val split, GPU-side augmentation
src/corruptions.py   GPU re-implementation of 6 CIFAR-C corruptions (noise, blur, contrast), severities 1-5
src/models/          MultiExitNet (segments + heads), ResNet-18, EE-ResNet-18, EE-ViT-Tiny
src/exits.py         exit scores, temperature scaling, ECE, offline routing, real early-exit inference
src/flops.py         per-exit FLOPs with torch FlopCounterMode
src/train.py         multi-exit training (equal-weight CE over exits), AMP, cosine schedule
src/evaluate.py      logits for clean and corrupted test sets, temperatures, latency benchmarks
kaggle/              Kaggle GPU job template and launcher
scripts/analyze.py   threshold sweeps, calibration, corruption and latency analysis, figures
results/             metrics, figures and summary.json produced by scripts/analyze.py
report/              interim report source and output
```

## Running

Training runs on Kaggle T4 GPUs; the kernel clones this repository at `HEAD`.

```bash
pip install -r requirements.txt
python kaggle/launch.py push resnets     # ResNet-18 and EE-ResNet-18 in parallel on 2 x T4
python kaggle/launch.py push vit         # EE-ViT-Tiny
python kaggle/launch.py status resnets
python kaggle/launch.py fetch resnets    # outputs to runs/resnets/
python kaggle/launch.py fetch vit
python scripts/analyze.py                # writes results/
python report/build_report.py            # writes report/Interim_Report.docx and .pdf
```

Local smoke test on CPU:

```bash
python -m src.train --model ee_resnet18 --out runs/smoke --device cpu --epochs 1 --limit 256
python -m src.evaluate --run runs/smoke --device cpu --no-corrupt --cpu-images 20
```
