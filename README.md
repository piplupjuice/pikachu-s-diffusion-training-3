# pikachu-s-diffusion-training-3
Zero shot medical image generation via Stein's Identity
## What is this?

It explores whether we can generate medical images
without training any neural network from scratch.

Core idea:
- Extract score functions from frozen CLIP via Stein's Identity
- Adapt to the medical domain using 100 real X-rays + RBF kernel
- Generate images via Langevin dynamics

No training. No fine-tuning. Just math.

---

## The Math

Score extraction via Stein's Identity:
∇ log p(x) = -E[∇f(x)] / E[f(x)]

Domain adaptation via RBF kernel:
K(x, xi) = exp(-||phi(x) - phi(xi)||^2 / 2*sigma^2)
alpha = (K + lambda*I)^{-1} * 1

Generation via Langevin dynamics:
x_{t+1} = x_t + step * score(x_t) + noise

---

## Experiments

### V1 — Pixel Space
- Method: CLIP score + kernel directly in pixel space
- Result: CLIP similarity 0.19 → 0.65 on avg.
- Limitation: Images noisy — CLIP gradients unstructured in pixel space

### V2 — Latent Space
- Method: CLIP score + kernel through frozen SD-VAE latent space
- Result: CLIP similarity 0.22 → 0.42 on avg.
- Limitation: SD-VAE trained on natural photos not medical images

---

## How to Run

1. Go to kaggle.com → New Notebook → Import either .ipynb file
2. Sidebar → + Add Data → search: chest-xray-pneumonia by Paul Mooney
3. Set Accelerator → GPU P100
4. Turn Internet ON
5. Press Run All

## Dataset

Name    : Chest X-Ray Images (Pneumonia)
Author  : Paul Mooney
Source  : kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Used for: Kernel adaptation only — NOT training
Images  : 100 out of 5,863 available
Labels  : None used

---

## Honest Limitations

1. Images not clinically usable yet
2. SD-VAE latent space not optimized for medical imaging
3. CLIP similarity is not FID — not a clinical quality metric
4. No radiologist evaluation done yet
5. just doing for nothing :3 

---
## Status

just a normal experiment from curiosity, nothing else, this intuition can be wrong ...

