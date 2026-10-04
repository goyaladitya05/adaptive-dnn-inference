# adaptive-dnn-inference

Adaptive Deep Neural Network Inference using Confidence-Based Early Exits (ICT-4442 Deep Learning project).

Images exit a network at an intermediate classifier once the prediction is confident enough, so easy inputs use less computation. We compare, on CIFAR-100:

1. ResNet-18 with full inference (baseline)
2. Confidence-based early-exit ResNet-18 (max softmax probability threshold)
3. Calibrated / entropy-based early-exit ResNet-18 (per-exit temperature scaling, entropy threshold)
4. Early-exit ViT-Tiny

and measure accuracy, FLOPs, real latency (GPU and CPU, batch 1) and behaviour under input corruptions.
