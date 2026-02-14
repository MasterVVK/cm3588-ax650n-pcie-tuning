# Benchmark Results

## Test Configuration

- **Runtime**: AXCL aarch64 (ax-llm)
- **Device**: AX650N (M5Stack Module LLM / AI-8850)
- **Host**: CM3588 NAS (RK3588, 32GB RAM)
- **PCIe**: Gen2 x1 (500 MB/s)
- **Driver**: AXCL V3.6.4
- **Kernel**: 6.1.118
- **Date**: 2026-02-09 — 2026-02-13

## LLM Inference — Multi-Model Comparison

### Decode Speed (tok/s)

| Model | Quant | Layers | Size | Default | Optimized | Speedup | Native (official) |
|-------|-------|-------:|-----:|--------:|----------:|--------:|------------------:|
| MiniCPM4-0.5B | W8A16 | 24 | 1.5 GB | 6-11 | **15-20** | +100% | 36 |
| SmolLM2-360M | W8A16 | 32 | 1.3 GB | 5-10 | **13-16** | +75% | 38.7 |
| Qwen3-0.6B | W8A16 | 28 | 1.0 GB | 7.1-7.5 | **10-12** | +50% | 19-20 |
| DeepSeek-R1-1.5B | W4A16 | 28 | 2.4 GB | 4.8-7.0 | **10.2-11.0** | +75% | 17.7 |
| DeepSeek-R1-1.5B | W8A16 | 28 | 3.2 GB | 4.0-5.2 | **7.6-8.6** | +95% | 17.7 |
| Qwen3-1.7B | W8A16 | 24 | 2.7 GB | 5.1-5.3 | **7.8-8.0** | +50% | 7.42 |
| Qwen2.5-7B | W4A16 | 28 | 5.2 GB | 3.7 | **4.4** | +19% | 4.8 |
| SmolLM3-3B | W8A16 | 36 | 4.6 GB | 2.6-3.2 | **4.3-4.4** | +50% | — |
| Qwen3-4B | W8A16 | 36 | 5.1 GB | 2.6-2.8 | **3.7** | +37% | — |

### TTFT (Time To First Token)

| Model | Default | Optimized | Speedup |
|-------|--------:|----------:|--------:|
| MiniCPM4-0.5B | 318-350 ms | **234-244 ms** | +30% |
| SmolLM2-360M | 347-373 ms | **285-304 ms** | +20% |
| Qwen3-0.6B | 488-578 ms | **391 ms** | +25% |
| DeepSeek-R1-1.5B | 509-661 ms | **380-432 ms** | +35% |
| Qwen3-1.7B | 541 ms | **447 ms** | +21% |
| SmolLM3-3B | 916-1043 ms | **708-735 ms** | +30% |
| Qwen3-4B | 1216 ms | **1110 ms** | +10% |

*Note: Qwen2.5-7B binary does not log TTFT.*

### Key Observations

- **MiniCPM4-0.5B** and **SmolLM2-360M** show the highest optimization gains at **+75-100%** — small, efficient architectures benefit enormously from reduced PCIe latency
- **Baseline (schedutil) is extremely unstable** — up to 2x variance (MiniCPM4: 6-11 tok/s). Optimization dramatically improves both speed AND stability
- **DeepSeek-R1-1.5B W4A16** reaches **11 tok/s** with optimization, making reasoning models practical on edge hardware
- **Qwen3-1.7B** optimized reaches **~108% of official native** (7.9 vs 7.42 tok/s) — PCIe overhead effectively eliminated for compute-bound models
- **Qwen2.5-7B** optimized reaches **92% of native** (4.4 vs 4.8 tok/s)
- Small models (<1B) achieve only 35-55% of native speed — PCIe Gen2 x1 is the bottleneck for fast token generators
- Larger models (>1.5B) are more compute-bound, approaching native speed as PCIe overhead becomes negligible
- **9 LLM configurations** tested across 7 model families (Qwen3, Qwen2.5, MiniCPM4, SmolLM2, SmolLM3, DeepSeek-R1)

### Cross-Platform Comparison (Qwen3-0.6B)

| Platform | PCIe | tok/s | TTFT | Notes |
|----------|------|------:|-----:|-------|
| AX650N native | — | 19-20 | — | No PCIe overhead |
| RPi5 + M.2 HAT | Gen2 x1 | ~13 | — | BCM2712 |
| **CM3588 (optimized)** | **Gen2 x1** | **10-12** | **391 ms** | **This project** |
| CM3588 (default) | Gen2 x1 | 7.1-7.5 | 488-578 ms | No optimization |

### Cross-Platform Comparison (SmolLM2-360M)

| Platform | PCIe | tok/s | Notes |
|----------|------|------:|-------|
| AX650N native | — | 38.7 | No PCIe overhead |
| RPi5 + M.2 HAT | Gen2 x1 | 20.8 | BCM2712 (54% of native) |
| **CM3588 (optimized)** | **Gen2 x1** | **12.2-14.0** | **This project (35% of native)** |
| CM3588 (default) | Gen2 x1 | 7.0-9.6 | No optimization |

### Qwen3-0.6B Detailed Results

#### Without Optimization (default after reboot)

| Run | TTFT (ms) | Decode (tok/s) |
|-----|-----------|----------------|
| 1 | 526 | 7.07 |
| 2 | 578 | 7.48 |
| 3 | 488 | 7.28 |

**Average: ~7.3 tok/s, TTFT ~531 ms**

#### With Optimization (IRQ affinity + performance governor)

| Run | TTFT (ms) | Decode (tok/s) |
|-----|-----------|----------------|
| 1 | 391 | 9.98 |
| 2 | — | 10.52 |
| 3 | — | 12.12 |

**Average: ~10.9 tok/s** (some run-to-run variance)

#### Improvement

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Decode speed (avg) | 7.3 tok/s | 10.9 tok/s | **+49%** |
| TTFT (avg) | 531 ms | 391 ms | **-26%** |
| Stability | Moderate variance | Improved | |

### MiniCPM4-0.5B Detailed Results

Native (AX650): 36 tok/s. Binary: main_axcl_aarch64 (same as Qwen3). 24 layers, W8A16, tokens_embed: 73448x1024.

#### Without Optimization

| Run | TTFT (ms) | Decode (tok/s) |
|-----|-----------|----------------|
| 1 | 318 | 9.40 |
| 2 | 350 | 8.50 |
| 3 | 333 | 8.50 |
| 4 (retest) | 423 | 6.17 |
| 5 (retest) | 305 | 10.61 |
| 6 (retest) | 362 | 8.49 |

**Baseline extremely unstable with schedutil: 6.17-10.61 tok/s (1.7x variance!)**

#### With Optimization

| Run | TTFT (ms) | Decode (tok/s) |
|-----|-----------|----------------|
| 1 | 234 | 18.40 |
| 2 | 244 | 15.40 |
| 3 | 240 | 16.80 |
| 4 (retest) | 228 | 16.73 |
| 5 (retest) | 240 | 14.60 |
| 6 (retest) | 214 | 19.84 |

**Speedup: +100% decode (median 8.5 → 16.7), +40% TTFT.** Optimized = 46% of native (16.7 vs 36). Optimization also dramatically improves stability.

### SmolLM2-360M-Instruct Detailed Results

Native (AX650): 38.7 tok/s, RPi5 PCIe: 20.8 tok/s. Binary: main_axcl_aarch64 (unique binary). 32 layers, W8A16, tokens_embed: 49152x960.

#### Without Optimization

| Run | TTFT (ms) | Decode (tok/s) |
|-----|-----------|----------------|
| 1 | 347 | 7.0 |
| 2 | 373 | 9.1 |
| 3 | 365 | 9.6 |
| 4 (retest) | 511 | 5.13 |
| 5 (retest) | 530 | 5.01 |
| 6 (retest) | 431 | 8.72 |

**Baseline extremely unstable with schedutil: 5.01-9.6 tok/s (1.9x variance!)**

#### With Optimization

| Run | TTFT (ms) | Decode (tok/s) |
|-----|-----------|----------------|
| 1 | 304 | 12.2 |
| 2 | 293 | 13.9 |
| 3 | 285 | 14.0 |
| 4 (retest) | 290 | 15.01 |
| 5 (retest) | 259 | 16.25 |
| 6 (retest) | 272 | 13.60 |

**Speedup: +75% decode (median 7.9 → 13.9), +35% TTFT.** Optimized = 36% of native (13.9 vs 38.7), 67% of RPi5 (13.9 vs 20.8). Optimization also dramatically reduces variance.

### DeepSeek-R1-Distill-Qwen-1.5B Detailed Results

Native (AX650): 17.68 tok/s. Reasoning model (generates `<think>` block). 28 layers, Qwen2 architecture.

#### W8A16 — Without Optimization

| Run | TTFT (ms) | Decode (tok/s) |
|-----|-----------|----------------|
| 1 | 553 | 5.12 |
| 2 | 661 | 4.19 |
| 3 | 541 | 5.15 |
| 4 (retest) | 688 | 4.16 |
| 5 (retest) | — | timeout |
| 6 (retest) | 672 | 3.99 |

#### W8A16 — With Optimization

| Run | TTFT (ms) | Decode (tok/s) |
|-----|-----------|----------------|
| 1 | 403 | 8.20 |
| 2 | 432 | 7.60 |
| 3 | 406 | 8.25 |
| 4 (retest) | 418 | 8.60 |
| 5 (retest) | 400 | 8.62 |
| 6 (retest) | 433 | 7.59 |

**W8A16 Speedup: +95% decode (median 4.2 → 8.2), +35% TTFT.** Optimized = 46% of native (8.2 vs 17.7).

#### W4A16 (GPTQ-Int4) — Without Optimization

| Run | TTFT (ms) | Decode (tok/s) |
|-----|-----------|----------------|
| 1 | 533 | 6.54 |
| 2 | 509 | 6.43 |
| 3 | 627 | 4.87 |
| 4 (retest) | 608 | 4.75 |
| 5 (retest) | 527 | 6.67 |
| 6 (retest) | 561 | 6.30 |

#### W4A16 (GPTQ-Int4) — With Optimization

| Run | TTFT (ms) | Decode (tok/s) |
|-----|-----------|----------------|
| 1 | 383 | 10.48 |
| 2 | 380 | 11.02 |
| 3 | 394 | 10.20 |

**W4A16 Speedup: +75% decode (median 6.4 → 10.5), +35% TTFT.** Optimized = 59% of native (10.5 vs 17.7). INT4 quantization is 28% faster than W8A16 with optimization.

### SmolLM3-3B Detailed Results

HuggingFace SmolLM3-3B. Binary: SmolLM2 main_axcl_aarch64 (compatible). 36 layers, W8A16, tokens_embed: 128256x2048. Embedding converted from .npy to .bfloat16.bin. Thinking disabled via tokenizer (enable_thinking=False).

#### Without Optimization

| Run | TTFT (ms) | Decode (tok/s) |
|-----|-----------|----------------|
| 1 | 1043 | 2.64 |
| 2 | 916 | 3.18 |
| 3 | 981 | 2.60 |

#### With Optimization

| Run | TTFT (ms) | Decode (tok/s) |
|-----|-----------|----------------|
| 1 | 735 | 4.30 |
| 2 | 708 | 4.34 |
| 3 | 726 | 4.38 |

**Speedup: +50% decode, +30% TTFT.** Similar to Qwen3-4B in both absolute speed and optimization gain, consistent with 3B+ parameter models being compute-bound.

## VLM (Vision-Language Model) Component Benchmarks

### Test Configuration

- **Models**: [SmolVLM2-256M-Video-Instruct](https://huggingface.co/AXERA-TECH/SmolVLM2-256M-Video-Instruct_Ax650) and [FastVLM-0.5B](https://huggingface.co/AXERA-TECH/FastVLM-0.5B) (pre-compiled for AX650)
- **Tool**: `axcl_run_model` (individual .axmodel components via PCIe)
- **Repeats**: 100 iterations, 5 warmup
- **Note**: No AXCL aarch64 binary available for end-to-end VLM inference. Only native (`main_ax650`) and AXCL x86 (`main_axcl_x86`) binaries exist. Component benchmarks show optimization effect on each VLM stage.

### SmolVLM2-256M-Video-Instruct

Native (AX650): 76.7 tok/s (image), 75.5 tok/s (video). 30 LLM layers, W8A16.

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Vision Encoder (512x512) | 99.115 | **98.352** | +0.8% |
| LLM Decoder Layer | 0.818 | **0.566** | +44.5% |
| LLM Post | 1.980 | **1.639** | +20.8% |

Estimated decode speed (30 layers + post):
- Default: ~38 tok/s (30×0.818 + 1.980 ≈ 26.5 ms/tok)
- Optimized: ~54 tok/s (30×0.566 + 1.639 ≈ 18.6 ms/tok)
- Native: 76.7 tok/s

### FastVLM-0.5B

Native (AX650): 34.8 tok/s. 24 LLM layers, W4A16 (Qwen2 architecture).

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Vision Encoder (512x512) | 45.507 | **44.642** | +1.9% |
| LLM Decoder Layer | 1.547 | **1.170** | +32.2% |
| LLM Post | 7.538 | **7.043** | +7.0% |

Estimated decode speed (24 layers + post):
- Default: ~22 tok/s (24×1.547 + 7.538 ≈ 44.7 ms/tok)
- Optimized: ~29 tok/s (24×1.170 + 7.043 ≈ 35.1 ms/tok)
- Native: 34.8 tok/s

### Analysis

VLM components follow the exact same optimization pattern as other models:
- **Vision encoders** (45-99ms): +1-2% — compute-bound, minimal PCIe overhead
- **LLM decoder layers** (0.6-1.5ms): **+32-45%** — PCIe latency dominant, huge optimization benefit
- **LLM post** (1.6-7.5ms): +7-21% — moderate benefit

The LLM decoder layers are called once per token and dominate decode speed. Optimization makes the biggest difference here, consistent with standalone LLM results. Estimated end-to-end decode speedup: +28-42%.

## MaxReadReq Experiments

These experiments were conducted to test whether PCIe register tuning could improve performance.

| MaxReadReq | Decode (tok/s) | Status |
|-----------|----------------|--------|
| 128 bytes | — | Driver crash |
| 256 bytes | — | Driver crash |
| **512 bytes (default)** | **9.11** | **Stable** |
| 4096 bytes | 6.38 | 30% slower |

**Conclusion**: MaxReadReq=512 is the only stable value for the AXCL driver.

## Vision Model Benchmarks

### Test Configuration

- **Models**: Downloaded from [HuggingFace AXERA-TECH](https://huggingface.co/AXERA-TECH)
- **Tool**: `axcl_run_model` (pure NPU inference, no pre/post-processing)
- **Repeats**: 100 iterations, 10 warmup
- **Input**: 640x640x3 (standard YOLO input)
- **Built from**: [axcl-samples](https://github.com/AXERA-TECH/axcl-samples)

### With vs Without Optimization

| Model | Task | Default avg (ms) | Optimized avg (ms) | Speedup | Optimized FPS |
|-------|------|------------------:|--------------------:|--------:|--------------:|
| YOLO11s | Detection | 3.994 | **3.548** | +12.6% | 282 |
| YOLOv8s | Detection | 4.206 | **3.891** | +7.5% | 257 |
| YOLO11s-Seg | Segmentation | 5.266 | **4.601** | +12.6% | 217 |
| YOLOv8s-Seg | Segmentation | 5.257 | **5.109** | +2.8% | 196 |
| YOLOv5s | Detection | 6.923 | **6.594** | +4.8% | 152 |
| YOLO26m | Detection | 9.472 | **9.042** | +4.5% | 111 |
| YOLOv8s-Pose | Pose | 11.812 | **11.259** | +4.7% | 89 |
| Depth-Anything-V2-S | Depth | 33.994 | **33.443** | +1.6% | 30 |

### YOLO26-Det (NPU 3-core, 640x640)

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Optimized FPS | Native (ms) |
|-------|------------------:|--------------------:|--------:|--------------:|------------:|
| YOLO26n-Det | 2.152 | **1.786** | +20.5% | 560 | 1.378 |
| YOLO26s-Det | 3.957 | **3.572** | +10.8% | 280 | 3.166 |
| YOLO26m-Det | 9.309 | **8.974** | +3.7% | 111 | 8.644 |
| YOLO26l-Det | 11.910 | **11.736** | +1.5% | 85 | 11.174 |
| YOLO26x-Det | 25.453 | **25.042** | +1.6% | 40 | 20.405 |

### YOLO26-Pose (NPU 3-core, 640x640)

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Optimized FPS | Native (ms) |
|-------|------------------:|--------------------:|--------:|--------------:|------------:|
| YOLO26n-Pose | 2.083 | **1.708** | +22.0% | 586 | 1.525 |
| YOLO26s-Pose | 4.071 | **3.717** | +9.5% | 269 | 3.528 |
| YOLO26m-Pose | 10.252 | **9.621** | +6.6% | 104 | 9.296 |
| YOLO26l-Pose | 12.899 | **12.159** | +6.1% | 82 | 11.963 |
| YOLO26x-Pose | 26.377 | **25.709** | +2.6% | 39 | 25.128 |

### YOLO26-Seg (NPU 3-core, 640x640)

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Optimized FPS | Native (ms) |
|-------|------------------:|--------------------:|--------:|--------------:|------------:|
| YOLO26n-Seg | 2.861 | **2.337** | +22.4% | 428 | 1.972 |
| YOLO26s-Seg | 5.614 | **5.035** | +11.5% | 199 | 4.703 |
| YOLO26m-Seg | 15.260 | **14.648** | +4.2% | 68 | 14.275 |
| YOLO26l-Seg | 17.563 | **17.270** | +1.7% | 58 | 16.675 |
| YOLO26x-Seg | 38.058 | **37.334** | +1.9% | 27 | 36.701 |

### Depth-Anything-3 (NPU 3-core)

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Native (ms) |
|-------|------------------:|--------------------:|--------:|------------:|
| DA3-small | 24.019 | **23.278** | +3.2% | 22.77 |
| DA3-base | 68.512 | **67.713** | +1.2% | 67.34 |

### Latency Stability (p95 - min)

| Model | Default (us) | Optimized (us) | Improvement |
|-------|------------:|---------------:|------------:|
| YOLO11s | 708 | 276 | 2.6x more stable |
| YOLOv8s | 604 | 197 | 3.1x more stable |
| YOLOv5s | 410 | 112 | 3.7x more stable |
| YOLO26m | 799 | 272 | 2.9x more stable |
| YOLOv8s-Seg | 374 | 339 | 1.1x |
| YOLO11s-Seg | 466 | 193 | 2.4x more stable |
| YOLOv8s-Pose | 560 | 228 | 2.5x more stable |
| Depth-Anything-V2-S | 672 | 249 | 2.7x more stable |

### Analysis

The optimization effect on vision models (2-13%) is smaller than on LLM (+50-100%) because:

1. **LLM decode** = hundreds of sequential small NPU calls + PCIe transfers. Each IRQ routing delay and frequency drop compounds across tokens.
2. **Vision inference** = single large NPU computation. The PCIe transfer overhead is a smaller fraction of total time.
3. **Stability** is the main benefit for vision: `schedutil` governor causes CPU frequency fluctuations between inference calls, adding 200-800us jitter. `performance` governor eliminates this.

For real-time video pipelines (30 FPS = 33ms budget), the stability improvement is critical: guaranteed latency matters more than average latency.

## LLM Methodology

- Each benchmark run: single prompt ("What is 2+2?" or similar short prompt)
- TTFT measured by ax-llm runtime
- Decode speed measured as average over full response generation
- "Cold" = first run after reboot (model loading from disk)
- "Warm" = subsequent runs (model in page cache)

## Classification Model Benchmarks

### Test Configuration

- **Models**: Exported from PyTorch `torch.hub` (torchvision v0.15.2), compiled with Pulsar2 5.1
- **Tool**: `axcl_run_model` (pure NPU inference)
- **Repeats**: 100 iterations, 10 warmup
- **Input**: 224x224x3 (standard ImageNet input, except SqueezeNet 227x227)
- **NPU mode**: 1 Core
- **Quantization**: INT8, MinMax calibration, 32 random samples

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Optimized FPS |
|-------|------------------:|--------------------:|--------:|--------------:|
| MobileNetV2 | 0.983 | **0.657** | +49.6% | 1523 |
| SqueezeNet1.1 | 0.786 | **0.768** | +2.3% | 1302 |
| ResNet18 | 1.963 | **1.435** | +36.8% | 697 |
| ResNet50 | 3.613 | **3.355** | +7.7% | 298 |

### Analysis

Classification models show the clearest correlation between model speed and optimization benefit:
- **MobileNetV2** (+50%): Very fast inference (~0.7ms) makes PCIe overhead dominant
- **ResNet18** (+37%): Still fast enough for significant benefit
- **ResNet50** (+8%): Longer inference dilutes the PCIe overhead effect
- **SqueezeNet1.1** (+2%): Despite fast inference, the model architecture may have different memory access patterns

## OCR Benchmarks — PPOCR_v5

### Test Configuration

- **Models**: PPOCR_v5 from [HuggingFace AXERA-TECH](https://huggingface.co/AXERA-TECH/PPOCR_v5)
- **Pipeline**: Detection (det) → Classification (cls) → Recognition (rec)
- **Runtime**: PyAXEngine (axengine 0.1.3) via AXCL
- **NPU mode**: 1 Core
- **Test images**: Chinese + English text (16 text regions detected)

### NPU Inference With vs Without Optimization

| Model | Task | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------|------------------:|--------------------:|--------:|
| det_npu1 | Text Detection | 29.38 | **28.98** | +1.4% |
| cls_npu1 | Text Direction | 0.759 | **0.445** | +70.6% |
| rec_npu1 | Text Recognition | 3.958 | **3.681** | +7.5% |

### Analysis

The OCR pipeline demonstrates the optimization effect pattern most clearly:
- **cls** (0.45ms): +71% — the fastest model benefits most
- **rec** (3.7ms): +8% — moderate inference time, moderate benefit
- **det** (29ms): +1% — longest inference, negligible benefit

Full OCR pipeline: ~1.5s per image (16 text regions, Chinese + English, 81-99% confidence).

## Video Transcode Benchmarks

### Test Configuration

- **Tool**: `axcl_sample_transcode` (from axcl-samples)
- **VPU**: AX650N hardware video encoder/decoder via PCIe
- **Input**: 1080p H.264 test video

### Results

| Task | Resolution | FPS | Optimization Effect |
|------|:----------:|----:|:-------------------:|
| Decode only | 1080p | 260 | None (0%) |
| Transcode H.264→H.265 | 1080p | ~28 | None (0%) |
| Transcode H.264→H.265 | 720p | ~28 | None (0%) |

### Analysis

PCIe optimization has **zero effect** on video transcode:
- VPU operations use DMA transfers, not IRQ-driven like NPU inference
- The bottleneck is the hardware encoder (~28 FPS), not PCIe bandwidth
- Decode is fast (260 FPS) — the encoder is the limiting factor
- Different `hwclk` settings also had no effect

This confirms the optimization specifically targets NPU inference workloads.

## Face Recognition — Insightface

### Test Configuration

- **Models**: [Insightface buffalo_l](https://huggingface.co/AXERA-TECH/Insightface) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 10 warmup

### With vs Without Optimization

| Model | Task | Input | Default avg (ms) | Optimized avg (ms) | Speedup | Optimized FPS |
|-------|------|-------|------------------:|--------------------:|--------:|--------------:|
| det_10g | Face Detection | 640x640 | 7.356 | **6.862** | +7.2% | 146 |
| 1k3d68 | 3D Landmarks | 192x192 | 3.045 | **2.639** | +15.4% | 379 |
| 2d106det | 2D Landmarks | 192x192 | 0.861 | **0.717** | +20.1% | 1395 |
| genderage | Gender/Age | 96x96 | 0.479 | **0.357** | +34.2% | 2801 |
| w600k_r50 | Face Embedding | 112x112 | 4.265 | **3.717** | +14.7% | 269 |

### Comparison with Official Numbers

| Model | Official (ms) | Our Optimized (ms) | Difference |
|-------|:------------:|:------------------:|:----------:|
| det_10g | 6.947 | 6.862 | -1% (faster) |
| genderage | 0.295 | 0.357 | +21% |
| w600k_r50 | 3.993 | 3.717 | -7% (faster) |

## Super-Resolution — Real-ESRGAN

### Test Configuration

- **Models**: [Real-ESRGAN x4](https://huggingface.co/AXERA-TECH/Real-ESRGAN) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Quantization**: w8a8 (INT8)

### With vs Without Optimization

| Model | Input→Output | Default avg (ms) | Optimized avg (ms) | Speedup | Official (ms) |
|-------|:------------:|------------------:|--------------------:|--------:|--------------:|
| realesrgan-x4 | 64→256 | 15.850 | **15.663** | +1.2% | 15 |
| realesrgan-x4-256 | 256→1024 | 476.199 | **475.425** | +0.2% | 440 |

### Analysis

Real-ESRGAN shows minimal optimization benefit — the models are compute-heavy (especially 256→1024 at 475ms). The 8% gap vs official numbers on the large model is due to PCIe bandwidth: upscaling 256→1024 transfers large tensors.

## Segment Anything — MobileSAM

### Test Configuration

- **Models**: [MobileSAM](https://huggingface.co/AXERA-TECH/MobileSAM) encoder + decoder (pre-compiled for AX650)
- **Input**: 1024x1024

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Official (ms) |
|-------|------------------:|--------------------:|--------:|--------------:|
| Encoder | 51.521 | **50.920** | +1.2% | 49.495 |
| Decoder | 10.585 | **10.349** | +2.3% | 9.930 |

## Zero-Shot Classification — SigLIP2

### Test Configuration

- **Models**: [siglip2-base-patch16-224](https://huggingface.co/AXERA-TECH/siglip2-base-patch16-224) (pre-compiled for AX650)
- **Quantization**: w8a16

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Official (ms) |
|-------|------------------:|--------------------:|--------:|--------------:|
| Vision encoder (224x224) | 11.482 | **11.369** | +1.0% | 11.1 |
| Text encoder | 5.003 | **4.567** | +9.6% | 4.56 |

## Additional Detection Models

### With vs Without Optimization

| Model | Source | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|--------|------------------:|--------------------:|--------:|
| RT-DETR | [AXERA-TECH](https://huggingface.co/AXERA-TECH/RT-DETR) | 9.515 | **9.346** | +1.8% |
| YOLO-World YOLO | [AXERA-TECH](https://huggingface.co/AXERA-TECH/YOLO-World-V2) | 9.707 | **9.252** | +4.9% |
| YOLO-World CLIP | [AXERA-TECH](https://huggingface.co/AXERA-TECH/YOLO-World-V2) | 3.321 | **3.049** | +8.9% |

## Speech Recognition — Whisper

### Test Configuration

- **Models**: [Whisper-tiny](https://huggingface.co/AXERA-TECH/Whisper) encoder + decoder (pre-compiled for AX650)
- **Tool**: `axcl_run_model` (pure NPU benchmark, not full pipeline)

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| Encoder | 21.268 | **21.132** | +0.6% |
| Decoder | 4.054 | **3.932** | +3.1% |

## Speech Recognition — SenseVoice

### Test Configuration

- **Models**: [SenseVoice](https://huggingface.co/AXERA-TECH/SenseVoice) (FunASR, supports zh/en/ja/ko/yue + emotion)
- **Tool**: `axcl_run_model`
- **Repeats**: 20 iterations, 3 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| SenseVoice (full) | 55.328 | **54.691** | +1.2% |
| SenseVoice (streaming) | 13.105 | **12.369** | +6.0% |

### Analysis

The streaming variant (13ms) benefits more from optimization (+6%) than the full model (55ms, +1%) — consistent with the inference-time pattern. SenseVoice is a 250MB model supporting 5 languages with emotion recognition.

## CLIP — MobileCLIP2

### Test Configuration

- **Models**: [MobileCLIP2](https://huggingface.co/AXERA-TECH/MobileCLIP) S0 and S4 (Apple, pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations (S0), 20 iterations (S4 image), 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Official (ms) |
|-------|------------------:|--------------------:|--------:|--------------:|
| S0 Image (256x256) | 8.626 | **8.485** | +1.7% | — |
| S4 Image (384x384) | 64.942 | **64.339** | +0.9% | 65.33 |
| S4 Text | 13.081 | **12.895** | +1.4% | 12.66 |

## TTS — CosyVoice3

### Test Configuration

- **Pipeline**: LLM (Qwen2 24-layer) → Flow Decoder (ODE 7 steps) → HiFi-GAN
- **Text**: Russian ("Искусственный интеллект открывает новые возможности для человечества.")
- **Binary**: Custom C++ via AXCL
- **3 runs per configuration**, averaged

### With vs Without Optimization

| Metric | Default | Optimized | Speedup |
|--------|--------:|----------:|--------:|
| TTFT | 125.3 ms | **108.3 ms** | +15.7% |
| LLM Decode | 13.94 tok/s | **16.34 tok/s** | +17.2% |
| RTF (Real-Time Factor) | 2.0-3.7x | **1.7-1.9x** | |

### Analysis

CosyVoice3 TTS optimization effect is moderate (+17% LLM decode) because:
- LLM decode benefits from optimization (sequential NPU calls)
- Token2Wav (ODE solver with 7 steps of ~100ms each) dominates total time and shows minimal benefit
- RTF > 1.0 means slower than real-time on PCIe Gen2 x1

## TTS — Kokoro

### Test Configuration

- **Models**: [kokoro.axera](https://huggingface.co/AXERA-TECH/kokoro.axera) — 3-part TTS pipeline (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 20 iterations (Part1, Part3), 100 iterations (Part2), 3 warmup
- **Note**: Full pipeline also includes ONNX model4 (harmonic simulator, runs on CPU). Only NPU parts benchmarked.

### With vs Without Optimization

| Model | CMM (MB) | Default avg (ms) | Optimized avg (ms) | Speedup | Native* (ms) |
|-------|:--------:|------------------:|--------------------:|--------:|-------------:|
| Part1 (encoder) | 87 | 17.573 | **16.829** | +4.4% | 22.1 |
| Part2 (decoder) | 22 | 9.958 | **9.655** | +3.1% | 17.4 |
| Part3 (vocoder) | 118 | 189.967 | **189.508** | +0.2% | 185.3 |

*Native numbers from Python pyaxengine (includes Python runtime overhead ~5-8ms). Part1 and Part2 are faster via C++ axcl_run_model than native Python.*

### Analysis

Kokoro TTS is a 3-part pipeline with RTF ~0.07 on native AX650N (14x real-time). Via PCIe:
- Part3 (vocoder, 190ms) dominates pipeline time and is fully compute-bound (+0.2%)
- Part1 (encoder, 17ms) and Part2 (decoder, 10ms) show modest benefit (+3-4%)
- Total NPU time: ~216ms per inference (vs ~225ms native Python) — C++ runtime is actually faster than native Python for light models
- PCIe overhead visible only on Part3: +2.3% vs native (189.5 vs 185.3ms)

## Optimization Effect Pattern

The speedup from PCIe optimization correlates inversely with inference time:

| Inference time | Model | Speedup |
|:-:|:-:|:-:|
| ~0.20 ms | LivePortrait stitching | **+57%** |
| ~0.24 ms | Zipformer decoder | **+80%** |
| ~0.27 ms | EdgeTAM prompt encoder | **+10%** |
| ~0.3 ms | Insightface genderage | **+34%** |
| ~0.33 ms | VoxCPM stop_predictor | **+67%** |
| ~0.34 ms | VoxCPM FSQ layer | **+38%** |
| ~0.34 ms | Zipformer joiner | **+93%** |
| ~0.35 ms | PPOCR_v5 cls (npu3) | **+18%** |
| ~0.45 ms | VoxCPM LocDiT part1 | **+53%** |
| ~0.45 ms | SmolVLM-256M LLM layer | **+56%** |
| < 0.5 ms | OCR classifier cls (npu1) | **+71%** |
| ~0.57 ms | SmolVLM2-256M LLM layer | **+45%** |
| ~0.7 ms | MobileNetV2/Insightface 2d106det | **+20-50%** |
| ~0.7 ms | EdgeTAM prompt mask | +4% |
| ~0.9 ms | Qwen2.5-0.5B LLM layer (Int4) | **+33%** |
| ~1.0 ms | VoxCPM base_lm MiniCPM layer | **+57%** |
| ~1.0 ms | InternVL2.5-1B LLM layer/SmolVLM2-500M LLM layer | **+25-62%** |
| ~1.1 ms | VoxCPM feat_encoder MiniCPM layer | **+44%** |
| ~1.1 ms | InternVL3-1B LLM layer | **+54%** |
| ~1.2 ms | InternVL3.5-1B LLM layer (Int4) | **+26%** |
| ~1.2 ms | FastVLM-0.5B LLM layer | **+32%** |
| ~1.4 ms | Qwen3-0.6B LLM layer (Int4) | **+24%** |
| ~1.5 ms | FastVLM-1.5B LLM layer (Int4) | +1% |
| ~1.4 ms | ResNet18 | **+37%** |
| ~1.4 ms | gtcrn (audio denoise) | +12% |
| ~1.5 ms | InternVL3.5-1B LLM layer (Qwen3) | **+26%** |
| ~1.6 ms | SmolVLM-256M LLM post | **+33%** |
| ~1.6 ms | SmolVLM2-256M LLM post | +21% |
| ~1.6 ms | SATRN decoder | **+37%** |
| ~1.7 ms | PPOCR_v5 rec (npu3) | **+18%** |
| ~1.8 ms | YOLO26n-Det | **+21%** |
| ~1.7 ms | YOLO26n-Pose | **+22%** |
| ~1.6 ms | DeepSeek-R1-1.5B LLM layer (Int4) | **+24%** |
| ~1.7 ms | Qwen2.5-1.5B LLM layer (Int4) | **+28%** |
| ~1.8 ms | HY-MT1.5 LLM layer (HuanYuan) | +12% |
| ~1.9 ms | InternVL3.5-2B LLM layer (Int4) | **+27%** |
| ~2.0 ms | Gemma-3-1B LLM layer | **+23%** |
| ~1.8 ms | Qwen3-Embedding layer | +12.5% |
| ~2.3 ms | YOLO26n-Seg | **+22%** |
| ~2.1 ms | Qwen3-VL-2B LLM layer (Int4) | **+19%** |
| ~2.5 ms | Qwen2.5-3B LLM layer (Int4) | +7% |
| ~2.6 ms | Qwen3-1.7B LLM layer (Int4) | +11% |
| ~2.6 ms | Qwen2.5-1.5B LLM layer (W8A16) | +15% |
| ~2.6 ms | Insightface 1k3d68 (3D landmarks) | +15% |
| ~2.4 ms | Qwen2.5-VL-3B LLM layer (Int4) | **+22%** |
| ~2.7 ms | SmolVLM2-500M Post/FastVLM-1.5B LLM layer | +11-16% |
| ~3.0 ms | Zipformer encoder/InternVL3-2B LLM layer | +5-19% |
| ~3.1 ms | CAM++ speaker embedding | +17% |
| ~3.0 ms | YOLO-World CLIP | +9% |
| ~3.2 ms | Qwen3-VL-2B LLM layer/InternVL3.5-2B LLM layer | +9% |
| ~3.4 ms | Qwen3-VL-4B LLM layer (Int4) | +8% |
| ~3.4 ms | Janus-Pro-1B LLM layer | +15% |
| ~3.5 ms | ResNet50 | +8% |
| ~3.9 ms | Qwen2.5-3B LLM layer (W8A16) | +15% |
| ~3.6 ms | YOLO11s/YOLO26s-Det/QR YOLO26n/YOLO11n | +2-12% |
| ~3.7 ms | YOLO26s-Pose/Insightface w600k_r50 | +10-15% |
| ~3.9 ms | 3D-Speaker ECAPA-TDNN | +3% |
| ~4.0 ms | QR DEIMv2-femto | +9% |
| ~4.0 ms | YOLOv8s detection | +10% |
| ~4.4 ms | cnclip ViT-L/14 text | +14% |
| ~4.6 ms | LibCLIP cnclip text | +10% |
| ~4.7 ms | YOLO11s-Seg | +2% |
| ~5.0 ms | YOLO26s-Seg/YOLOv8s-Seg | +4-12% |
| ~5.0 ms | MiMo-VL-7B LLM layer (Int4) | +14% |
| ~5.3 ms | MiniCPM-V-4 LLM layer (LLaMA) | +12% |
| ~5.2 ms | EdgeTAM mask decoder | +3% |
| ~5.5 ms | 3D-Speaker Res2NetV2 | +1% |
| ~5.8 ms | SileroVAD | +9% |
| ~6.3 ms | DeepSeek-R1-7B LLM layer (Int4) | +10% |
| ~5.8 ms | CLIP ViT-L/14 text | +10% |
| ~7.0 ms | FastVLM-0.5B LLM post | +7% |
| ~7.0 ms | InternVL3-1B/InternVL2.5-1B LLM post | +6-8% |
| ~7.3 ms | Qwen3-4B-2507-Int4 LLM layer | +5% |
| ~7 ms | YOLOv5s/Insightface det | +5-7% |
| ~7.5 ms | LivePortrait motion | +9% |
| ~8.5 ms | MobileCLIP2-S0 image | +2% |
| ~8.1 ms | InternVL3.5-1B LLM post | +5% |
| ~9 ms | RT-DETR/YOLO-World YOLO/YOLO26m-Det | +2-5% |
| ~9.2 ms | MiniCPM-V-4 LLM post | +3% |
| ~9.7 ms | Kokoro Part2 (TTS decoder) | +3% |
| ~9.6 ms | YOLO26m-Pose | +7% |
| ~10 ms | YOLOv5s-Seg | +2% |
| ~10.4 ms | Janus-Pro-1B LLM post | +7% |
| ~10.4 ms | MixFormerV2 (tracking) | +3% |
| ~11.7 ms | Qwen2.5-1.5B LLM post (W8A16/Int4) | +2-4% |
| ~11.5 ms | FastVLM-1.5B LLM post | +7% |
| ~10.4 ms | ESPCN x2 2K | +2% |
| ~11.3 ms | YOLOv8s-Pose | +0.2% |
| ~11 ms | FG-CLIP text/SigLIP2 vision | +1-5% |
| ~11.7 ms | YOLO26l-Det | +1% |
| ~12.2 ms | YOLO26l-Pose | +6% |
| ~12.4 ms | HY-MT1.5 LLM post | +4% |
| ~12.4 ms | SenseVoice streaming | +6% |
| ~13 ms | YOLOv7-Face/DeepLabv3Plus/jina-clip text | +1-4% |
| ~15.5 ms | Gemma-3-1B LLM post | +5% |
| ~15.5 ms | Qwen2.5-VL-3B LLM post | +5% |
| ~15.7 ms | Qwen3-VL-2B LLM post | +2% |
| ~16 ms | RealESRGAN-x2 (CodeFormer) | +2% |
| ~17 ms | Kokoro Part1 (TTS encoder) | +4% |
| ~18 ms | PPOCR_v5 det (npu3) | +4% |
| ~18.8 ms | Qwen3-VL-4B-Int4 LLM post | +2% |
| ~18.8 ms | Qwen3-4B-2507-Int4 LLM post | +2% |
| ~20 ms | LivePortrait feature | +3% |
| ~21 ms | Whisper encoder/RAFT-stereo | ~0-1% |
| ~22 ms | ESPCN x2 | +1% |
| ~23 ms | Depth-Anything-3 small | +3% |
| ~23 ms | SigLIP-so400m text | +3% |
| ~24 ms | FireRedASR decoder/EdgeTAM image encoder | +1-3% |
| ~26 ms | DeepSeek-R1-7B LLM post (Int4) | +3% |
| ~25 ms | YOLO11x/YOLO11x-Pose | +0.4-3% |
| ~25 ms | YOLO26x-Det/YOLO26x-Pose | +2-3% |
| ~28 ms | SuperPoint | +1% |
| ~30 ms | MiMo-VL-7B LLM post (Int4) | +1% |
| ~29 ms | OCR detector (det)/YOLOv5l-Face | +1-2% |
| ~35 ms | YOLO11x-Seg/bge-small-en | +1-2% |
| ~37 ms | YOLO26x-Seg | +2% |
| ~43 ms | DEIMv2 DINOv3-S | +1% |
| ~45 ms | FastVLM-0.5B/1.5B vision encoder | +1-2% |
| ~51 ms | MobileSAM encoder | +1% |
| ~55 ms | SenseVoice (full) | +1% |
| ~65 ms | MobileCLIP2-S4 image | +1% |
| ~68 ms | Depth-Anything-3 base | +1% |
| ~70 ms | CLIP ViT-L/14 image | +1% |
| ~86 ms | Qwen3-VL-4B vision (u8) | +0.4% |
| ~89 ms | LibCLIP cnclip vision | +0.8% |
| ~92 ms | centerpoint/bevformer/MeloTTS | +0.5-0.7% |
| ~99 ms | SmolVLM2/SmolVLM-256M vision | +0.7-1% |
| ~113 ms | cnclip ViT-L/14 vision | +0.7% |
| ~107 ms | RMBG-1.4 (background removal) | +1% |
| ~113 ms | RAFT-stereo 384x1280 | ~0% |
| ~129 ms | FG-CLIP image encoder | +0.4% |
| ~143 ms | Janus-Pro-1B vision/IGEV++ | ~0-0.5% |
| ~158 ms | Qwen3-VL-2B vision | +0.4% |
| ~162 ms | FireRedASR encoder | +0.5% |
| ~168 ms | SigLIP-so400m vision | +0.5% |
| ~190 ms | Kokoro Part3 (TTS vocoder) | +0.2% |
| ~210 ms | RIFE x2 720p (frame interp) | +0.4% |
| ~233 ms | LivePortrait spade | +0.3% |
| ~309 ms | EDSR baseline x2 2K | +0.2% |
| ~357 ms | InternVL2.5-1B/InternVL3-1B/InternVL3.5-1B vision | +0.1-0.2% |
| ~383 ms | DeOldify (colorization) | +0.2% |
| ~426 ms | mel_band_roformer (music sep) | +0.2% |
| ~445 ms | CodeFormer (face restoration) | +0.1% |
| ~475 ms | Real-ESRGAN 256→1024 | +0.2% |
| ~560 ms | Qwen2.5-VL-3B vision (392x392) | +0.2% |
| ~581 ms | MiniCPM-V-4 SigLIP vision | +0.1% |
| ~498 ms | DeOldify artistic | +0.2% |
| ~597 ms | jina-clip-v2 image encoder | +0.2% |
| ~694 ms | EDSR baseline x2 | +0.1% |

**Why?** Each NPU inference involves PCIe round-trip overhead (~0.3ms for IRQ handling + data transfer). For fast models, this overhead is a significant fraction of total time. Moving IRQ to a faster CPU core (A76 @ 2.3 GHz vs A55 @ 1.8 GHz) reduces this overhead, and the `performance` governor eliminates frequency scaling delays between calls.

For LLM inference, the effect is even more dramatic (+50-100%) because each token requires hundreds of sequential small NPU calls, each incurring PCIe overhead. Smaller, more efficient LLM architectures (MiniCPM4, SmolLM2) show the highest gains. VLM decoder layers show +15-62% improvement — consistent with the LLM pattern.

Zipformer joiner at **+93%** is the absolute record — beating OCR classifier (+71%) as the previous champion. The sub-0.5ms models consistently show the most dramatic speedups, confirming that PCIe round-trip latency is the dominant factor for ultra-fast inference.

**190+ models tested** across 40+ categories confirm this pattern holds universally. For LLM, 11 configurations across 9 model families from 0.36B to 7B were tested, all showing significant speedup (+19% to +100%). VLM component benchmarks add 11 model families (SmolVLM2, FastVLM-0.5B/1.5B, SmolVLM, InternVL2.5, InternVL3, InternVL3.5, Janus-Pro, Qwen3-VL, Qwen2.5-VL, MiniCPM-V-4). Translation LLM (HY-MT1.5, HuanYuan architecture), speaker embedding (CAM++), TTS (CosyVoice3, Kokoro, MeloTTS, VoxCPM), portrait animation (LivePortrait), streaming ASR (Zipformer, FireRedASR), super-resolution, 3D detection, and VAD provide additional data points.

## Stereo Depth Estimation

### Test Configuration

- **Models**: [IGEV++](https://huggingface.co/AXERA-TECH/IGEV-plusplus) and [RAFT-stereo](https://huggingface.co/AXERA-TECH/RAFT-stereo) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 10 iterations, 3 warmup

### With vs Without Optimization

| Model | Input | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|-------|------------------:|--------------------:|--------:|
| RAFT-stereo | 256x640 | 21.19 | **21.28** | ~0% |
| IGEV++ (RTIGEV) | 480x640 | 143.40 | **143.06** | ~0% |
| RAFT-stereo | 384x1280 | 112.55 | **112.40** | ~0% |

### Analysis

Stereo depth models show virtually no optimization benefit — they are heavily compute-bound (20-143ms). PCIe round-trip overhead (~0.3ms) is negligible compared to total inference time.

## Video Segmentation — EdgeTAM

### Test Configuration

- **Models**: [EdgeTAM](https://huggingface.co/AXERA-TECH/EdgeTAM) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| Prompt encoder | 0.297 | **0.270** | +10.0% |
| Prompt mask encoder | 0.765 | **0.732** | +4.3% |
| Mask decoder | 5.338 | **5.184** | +2.9% |
| Image encoder | 23.88 | **23.73** | +0.6% |

### Analysis

EdgeTAM components follow the same speedup-vs-inference-time pattern. The tiny prompt encoder (0.27ms) benefits most. The image encoder (24ms) benefits least.

## Speaker Identification — 3D-Speaker

### Test Configuration

- **Models**: [3D-Speaker](https://huggingface.co/AXERA-TECH/3D-Speaker) ECAPA-TDNN and Res2NetV2 (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| ECAPA-TDNN | 4.006 | **3.889** | +3.0% |
| Res2NetV2 | 5.534 | **5.459** | +1.4% |

### Comparison with Official Numbers (Native)

| Model | Official native (ms) | Our AXCL optimized (ms) | PCIe overhead |
|-------|---------------------:|------------------------:|--------------:|
| IGEV++ (RTIGEV) | 139.80 | 143.06 | +2.3% |
| RAFT-stereo 256x640 | 20.9 | 21.28 | +1.8% |
| EdgeTAM image encoder | 22.35 | 23.73 | +6.2% |
| EdgeTAM mask decoder | 4.73 | 5.18 | +9.5% |
| EdgeTAM prompt encoder | 0.055 | 0.270 | +391% |
| EdgeTAM prompt mask | 0.457 | 0.732 | +60% |
| 3D-Speaker Res2NetV2 | 5.09 | 5.46 | +7.3% |
| 3D-Speaker ECAPA-TDNN | 7.37* | 3.89 | faster (!) |

*Note: ECAPA-TDNN official number is 7.37ms from Python pyaxengine. Our test uses C++ `axcl_run_model` which has lower overhead. The official test likely includes Python runtime costs.*

Note: EdgeTAM prompt encoder shows 5x PCIe overhead (0.055 vs 0.27ms) — for extremely fast models (<0.1ms native), the PCIe round-trip latency completely dominates.

## Audio Denoising — GTCRN

### Test Configuration

- **Models**: [gtcrn.axera](https://huggingface.co/AXERA-TECH/gtcrn.axera) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup
- **NPU mode**: 1 Core

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| gtcrn | 1.607 | **1.434** | +12.1% |

### Analysis

GTCRN at 1.4ms inference time shows +12% speedup — consistent with the optimization pattern for sub-2ms models. Note: GTCRN runs on 1 NPU core (vs 3 cores for most models), making PCIe overhead a larger fraction of total time.

## Video Frame Interpolation — RIFE

### Test Configuration

- **Models**: [RIFE.axera](https://huggingface.co/AXERA-TECH/RIFE.axera) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 20 iterations, 3 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Native (ms) |
|-------|------------------:|--------------------:|--------:|------------:|
| RIFE x2 720p | 210.522 | **209.761** | +0.4% | ~200 |
| RIFE x2 1080p | — | — | CMM overflow |
| RIFE x2 4K | — | — | CMM overflow |

### Analysis

RIFE 720p at 210ms is compute-bound — optimization has negligible effect. The 1080p model (493 MB CMM) and 4K model (2.4 GB CMM) fail to load via PCIe due to CMM memory constraints. At ~5 FPS (720p), RIFE is not real-time on AX650N.

## Object Tracking — MixFormerV2

### Test Configuration

- **Models**: [MixFormerV2](https://huggingface.co/AXERA-TECH/MixFormerV2) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Optimized FPS |
|-------|------------------:|--------------------:|--------:|--------------:|
| MixFormerV2 | 10.745 | **10.416** | +3.2% | 96 |

## Face Restoration — CodeFormer

### Test Configuration

- **Models**: [CodeFormer](https://huggingface.co/AXERA-TECH/CodeFormer) pipeline: face detection → restoration → upscale (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Task | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------|------------------:|--------------------:|--------:|
| YOLOv5l-Face | Face Detection | 29.566 | **28.942** | +2.2% |
| RealESRGAN-x2 | Face Upscale | 16.168 | **15.843** | +2.1% |
| CodeFormer | Face Restoration | 444.721 | **444.093** | +0.1% |

### Analysis

The CodeFormer pipeline shows the optimization pattern clearly: the lighter detection (30ms) and upscaling (16ms) components benefit slightly, while the heavy restoration model (445ms) is fully compute-bound.

## Photo Colorization — DeOldify

### Test Configuration

- **Models**: [DeOldify](https://huggingface.co/AXERA-TECH/DeOldify) artistic and stable variants (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| Colorize Stable | 383.607 | **382.969** | +0.2% |
| Colorize Artistic | 498.415 | **497.663** | +0.2% |

### Analysis

Both DeOldify variants are heavily compute-bound (383-498ms), so PCIe optimization has negligible effect (~0.2%).

## Background Removal — RMBG-1.4

### Test Configuration

- **Models**: [RMBG-1.4](https://huggingface.co/AXERA-TECH/RMBG-1.4) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| RMBG-1.4 | 107.205 | **106.514** | +0.6% |

## Semantic Segmentation — DeepLabv3Plus

### Test Configuration

- **Models**: [DeepLabv3Plus](https://huggingface.co/AXERA-TECH/DeepLabv3Plus) with MobileNet backbone (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Optimized FPS |
|-------|------------------:|--------------------:|--------:|--------------:|
| DeepLabv3Plus-MobileNet | 13.827 | **13.244** | +4.4% | 76 |

## Keypoint Detection — SuperPoint

### Test Configuration

- **Models**: [SuperPoint](https://huggingface.co/AXERA-TECH/SuperPoint) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Optimized FPS |
|-------|------------------:|--------------------:|--------:|--------------:|
| SuperPoint | 28.052 | **27.838** | +0.8% | 36 |

## QR Code Detection

### Test Configuration

- **Models**: [QRCode_det](https://huggingface.co/AXERA-TECH/QRCode_det) — multiple detector architectures trained for QR code detection (pre-compiled for AX650, npu1)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Optimized FPS |
|-------|------------------:|--------------------:|--------:|--------------:|
| YOLO26n | 4.083 | **3.634** | +12.4% | 275 |
| YOLO11n | 4.259 | **3.803** | +12.0% | 263 |
| DEIMv2-femto | 4.377 | **4.003** | +9.3% | 250 |

### Analysis

QR code detectors are compact nano-sized models (~4ms inference), making them good beneficiaries of PCIe optimization (+9-12%). YOLO26n and YOLO11n show nearly identical speedup.

## Additional Face Detection

### With vs Without Optimization

| Model | Source | Default avg (ms) | Optimized avg (ms) | Speedup | Optimized FPS |
|-------|--------|------------------:|--------------------:|--------:|--------------:|
| YOLOv7-Face | [AXERA-TECH](https://huggingface.co/AXERA-TECH/YOLOv7-Face) | 12.933 | **12.663** | +2.1% | 79 |

## Additional Detection — DEIMv2

### With vs Without Optimization

| Model | Source | Default avg (ms) | Optimized avg (ms) | Speedup | Optimized FPS |
|-------|--------|------------------:|--------------------:|--------:|--------------:|
| DEIMv2 DINOv3-S | [AXERA-TECH](https://huggingface.co/AXERA-TECH/DEIMv2) | 43.054 | **42.424** | +1.5% | 24 |

## Scene Text Recognition — SATRN

### Test Configuration

- **Models**: [SATRN](https://huggingface.co/AXERA-TECH/satrn) (backbone+encoder + decoder, pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Native (ms) |
|-------|------------------:|--------------------:|--------:|------------:|
| backbone+encoder | 7.431 | **7.347** | +1.1% | 6.085 |
| decoder | 2.170 | **1.582** | **+37.2%** | 1.384 |

### Analysis

SATRN decoder at 2.17ms shows **+37% speedup** — among the highest for any vision model. This tiny decoder is extremely sensitive to PCIe latency. The backbone+encoder (7.3ms) shows only +1%, consistent with the inference-time pattern.

## Embedding / RAG

### Test Configuration

- **Models**: [bge-small-en-v1.5](https://huggingface.co/AXERA-TECH/bge-small-en-v1.5) and [Qwen3-Embedding-0.6B](https://huggingface.co/AXERA-TECH/Qwen3-Embedding-0.6B) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Native (ms) |
|-------|------------------:|--------------------:|--------:|------------:|
| bge-small-en-v1.5 (batch 1) | 35.252 | **34.744** | +1.5% | 32.4 |
| bge-small-en-v1.5 (batch 2) | 63.584 | **63.152** | +0.7% | — |
| Qwen3-Embedding Layer (×28) | 2.025 | **1.800** | +12.5% | — |
| Qwen3-Embedding Post | 8.672 | **8.088** | +7.2% | — |

### Analysis

Qwen3-Embedding-0.6B has the same architecture as Qwen3-0.6B LLM (28 layers, W8A16). Individual decoder layers show +12.5% — consistent with the LLM pattern. Estimated full embedding: 28×1.80 + 8.09 ≈ 58.5ms (optimized) vs 28×2.03 + 8.67 ≈ 65.5ms (baseline). bge-small at 35ms is compute-bound with minimal PCIe overhead.

## CLIP — FG-CLIP, jina-clip-v2, LibCLIP

### Test Configuration

- **Models**: [FG-CLIP](https://huggingface.co/AXERA-TECH/FG-CLIP) (Qihoo 360), [jina-clip-v2](https://huggingface.co/AXERA-TECH/jina-clip-v2) (Jina AI), [LibCLIP/cnclip](https://huggingface.co/AXERA-TECH/LibCLIP) (Chinese CLIP ViT-L/14-336px)
- **Tool**: `axcl_run_model`
- **Repeats**: 50-100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Official (ms) |
|-------|------------------:|--------------------:|--------:|--------------:|
| LibCLIP cnclip text | 5.036 | **4.579** | +10.0% | 4.576 |
| FG-CLIP text encoder | 11.668 | **11.077** | +5.1% | 10.817 |
| jina-clip-v2 text encoder | 15.308 | **14.860** | +2.9% | 15.482 |
| LibCLIP cnclip vision | 89.439 | **88.763** | +0.8% | 88.475 |
| FG-CLIP image encoder | 129.194 | **128.688** | +0.4% | 125.197 |
| jina-clip-v2 image encoder | 597.219 | **596.199** | +0.2% | 592.231 |

### Analysis

CLIP models clearly demonstrate the optimization pattern: text encoders (5-15ms) benefit more than image encoders (89-597ms). Notable: LibCLIP text at 4.579ms exactly matches official native (4.576ms). jina-clip-v2 text (14.86ms) is actually faster than official native (15.48ms). Image encoders are compute-bound with minimal PCIe overhead.

## Music Source Separation — MelBandRoformer

### Test Configuration

- **Models**: [mel_band_roformer](https://huggingface.co/AXERA-TECH/mel_band_roformer) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 20 iterations, 3 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| mel_band_roformer | 426.329 | **425.619** | +0.2% |

### Analysis

At 426ms, mel_band_roformer is fully compute-bound — PCIe optimization has negligible effect. This model separates music into stems (bass, drums, vocals). Processing is offline (not real-time), so absolute speed matters more than latency consistency.

## Super-Resolution — EDSR & ESPCN

### Test Configuration

- **Models**: [SuperResolution](https://huggingface.co/AXERA-TECH/SuperResolution) — EDSR baseline and ESPCN (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 10 warmup (EDSR: 50 iterations, 5 warmup)

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Official (ms) |
|-------|------------------:|--------------------:|--------:|--------------:|
| ESPCN x2 | 22.705 | **22.411** | +1.3% | ~22 |
| ESPCN x2 2K | 10.664 | **10.418** | +2.4% | — |
| EDSR baseline x2 | 694.676 | **693.932** | +0.1% | ~800 |
| EDSR baseline x2 2K | 309.991 | **309.312** | +0.2% | — |

### Analysis

ESPCN is a lightweight SR model (~22ms) showing modest optimization benefit (+1-2%). EDSR at 694ms is heavily compute-bound (+0.1%). The "2K" variants use a different input resolution. EDSR official number (~800ms) is from Python pyaxengine, which adds runtime overhead — our C++ axcl_run_model is ~13% faster.

## 3D Object Detection — CenterPoint & BEVFormer

### Test Configuration

- **Models**: [CenterPoint](https://huggingface.co/AXERA-TECH/centerpoint) (3D LiDAR) and [BEVFormer](https://huggingface.co/AXERA-TECH/bevformer) (multi-camera BEV) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 20 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Native NPU3 (ms) |
|-------|------------------:|--------------------:|--------:|------------------:|
| CenterPoint-Pillar | 92.915 | **92.263** | +0.7% | 88.334 |
| BEVFormer-Tiny | 92.581 | **92.112** | +0.5% | 91.209 |

### Analysis

Both autonomous driving models are compute-bound at ~92ms. CenterPoint PCIe overhead: 4.4% vs native. BEVFormer: 1.0% overhead — remarkable given PCIe Gen2 x1 limitation. These models run at ~11 FPS, suitable for autonomous driving data processing.

## TTS — MeloTTS Decoder

### Test Configuration

- **Models**: [MeloTTS](https://huggingface.co/AXERA-TECH/MeloTTS) decoder (zh, en, jp variants) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 50 iterations, 5 warmup
- **Note**: MeloTTS has encoder (runs on ONNX) + decoder (on NPU). Only decoder benchmarked.

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| decoder-zh (Chinese) | 92.497 | **91.977** | +0.6% |
| decoder-en (English) | 92.732 | **92.115** | +0.7% |
| decoder-jp (Japanese) | 92.643 | **92.023** | +0.7% |

### Analysis

All MeloTTS decoders have nearly identical inference time (~92ms) — the decoder architecture is the same, only weights differ. At this latency, optimization has negligible effect (+0.7%). Official RTF: 0.125 (8x real-time on native AX650). Via PCIe, the decoder alone adds ~92ms per chunk.

## CLIP — OpenAI ViT-L/14-336px

### Test Configuration

- **Models**: [CLIP ViT-L/14-336px](https://huggingface.co/AXERA-TECH/clip) (OpenAI, pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 50 iterations (text), 20 iterations (image), 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| Text encoder | 6.386 | **5.821** | +9.7% |
| Image encoder | 71.108 | **70.283** | +1.2% |

### Analysis

CLIP text encoder at ~6ms benefits from optimization (+10%), consistent with the text-encoder pattern seen in SigLIP2, LibCLIP, and FG-CLIP. Image encoder at ~71ms is compute-bound with minimal benefit. This is the original OpenAI CLIP ViT-L/14 at 336px resolution.

## Zero-Shot — SigLIP-so400m-patch14-384

### Test Configuration

- **Models**: [siglip-so400m-patch14-384](https://huggingface.co/AXERA-TECH/siglip-so400m-patch14-384) (Google SigLIP 400M, pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 20 iterations, 3 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| Vision encoder (384x384) | 168.935 | **168.173** | +0.5% |
| Text encoder | 23.512 | **22.882** | +2.8% |

### Analysis

SigLIP-so400m is a much larger model than the previously tested SigLIP2-base (168ms vs 11ms for vision, 23ms vs 5ms for text). The vision encoder is heavily compute-bound. Text encoder shows +2.8% — lower than CLIP ViT-L/14 text (+10%) because it's 4x slower (23ms vs 6ms), making PCIe overhead a smaller fraction.

## VLM — SmolVLM-256M-Instruct (Component Benchmarks)

### Test Configuration

- **Models**: [SmolVLM-256M-Instruct](https://huggingface.co/AXERA-TECH/SmolVLM-256M-Instruct) (HuggingFace, pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Note**: Non-video version of SmolVLM2-256M. Same architecture: 30 LLM layers + vision encoder.

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Vision Encoder | 99.263 | **98.577** | +0.7% |
| LLM Decoder Layer | 0.700 | **0.450** | **+55.6%** |
| LLM Post | 2.110 | **1.582** | +33.4% |

Estimated decode speed (30 layers + post):
- Default: ~43 tok/s (30×0.700 + 2.110 ≈ 23.1 ms/tok)
- Optimized: ~66 tok/s (30×0.450 + 1.582 ≈ 15.1 ms/tok)
- Native: 80 tok/s

### Analysis

SmolVLM-256M (non-video) shows +56% on LLM layers — higher than SmolVLM2-256M (+45%). The smaller layer size (4.0MB vs 4.1MB) means even faster inference (0.45ms vs 0.57ms optimized), making PCIe overhead a larger fraction. Optimized reaches 83% of native speed.

## VLM — InternVL2.5-1B (Component Benchmarks)

### Test Configuration

- **Models**: [InternVL2.5-1B](https://huggingface.co/AXERA-TECH/InternVL2_5-1B) (OpenGVLab, pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Note**: 24 Qwen2 LLM layers, ViT vision encoder. Official native: 32 tok/s.

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Vision Encoder (448x448) | 357.773 | **356.981** | +0.2% |
| LLM Decoder Layer (Qwen2) | 1.648 | **1.020** | **+61.6%** |
| LLM Post | 7.533 | **7.098** | +6.1% |

Estimated decode speed (24 layers + post):
- Default: ~21 tok/s (24×1.648 + 7.533 ≈ 47.1 ms/tok)
- Optimized: ~32 tok/s (24×1.020 + 7.098 ≈ 31.6 ms/tok)
- **Native: 32 tok/s** — optimization eliminates PCIe overhead!

### Analysis

InternVL2.5-1B is the first VLM where optimized performance **matches official native speed** (32 tok/s). The 1B Qwen2 LLM layers at 1.0ms are in the sweet spot where optimization (+62%) maximally reduces PCIe overhead. Vision encoder at 358ms is fully compute-bound. This demonstrates that for compute-heavy LLM layers (~1ms per layer), optimization can completely compensate for PCIe Gen2 x1 bandwidth limitation.

## YOLO11 / YOLOv8 / YOLOv5-Seg — Detection, Pose, Segmentation

### Test Configuration

- **Models**: [YOLO11](https://huggingface.co/AXERA-TECH/YOLO11), [YOLO11-Pose](https://huggingface.co/AXERA-TECH/YOLO11-Pose), [YOLO11-Seg](https://huggingface.co/AXERA-TECH/YOLO11-Seg), [YOLOv8](https://huggingface.co/AXERA-TECH/YOLOv8), [YOLOv8-Pose](https://huggingface.co/AXERA-TECH/YOLOv8-Pose), [YOLOv8-Seg](https://huggingface.co/AXERA-TECH/YOLOv8-Seg), [YOLOv5-Seg](https://huggingface.co/AXERA-TECH/YOLOv5-Seg)
- **Tool**: `axcl_run_model`
- **Repeats**: 10 iterations, 3 warmup
- **Input**: 640x640

### With vs Without Optimization

| Model | Task | Default avg (ms) | Optimized avg (ms) | Speedup | Native (ms) |
|-------|------|------------------:|--------------------:|--------:|------------:|
| YOLO11s | Detection | 3.663 | **3.601** | +1.7% | — |
| YOLO11x | Detection | 25.857 | **25.191** | +2.6% | 25 |
| YOLOv8s | Detection | 4.454 | **4.003** | +10.1% | 3.6 |
| YOLO11x-Pose | Pose est | 25.652 | **25.548** | +0.4% | 25 |
| YOLOv8s-Pose | Pose est | 11.299 | **11.276** | +0.2% | 10.97 |
| YOLO11s-Seg | Segmentation | 4.760 | **4.676** | +1.8% | — |
| YOLO11x-Seg | Segmentation | 35.507 | **35.148** | +1.0% | 34 |
| YOLOv8s-Seg | Segmentation | 5.249 | **5.023** | +4.3% | 4.6 |
| YOLOv5s-Seg | Segmentation | 10.081 | **9.861** | +2.2% | 9.55 |

### Analysis

Large models (YOLO11x at 25ms) reach **99% of native speed** — PCIe overhead is negligible for compute-heavy models. YOLOv8s detection shows +10% — at 4ms it benefits more from reduced PCIe latency. YOLO11x-Pose at +0.4% suggests its architecture is pure compute-bound at this inference time. YOLOv5s-Seg reaches 97% of native.

## OCR — PPOCR_v5 (npu3 3-core)

### Test Configuration

- **Models**: [PPOCR_v5](https://huggingface.co/AXERA-TECH/PPOCR_v5) — det/cls/rec, npu3 variant (3 NPU cores)
- **Tool**: `axcl_run_model`
- **Repeats**: 10 iterations, 3 warmup

### With vs Without Optimization

| Model | Task | Default avg (ms) | Optimized avg (ms) | Speedup | Native (ms) |
|-------|------|------------------:|--------------------:|--------:|------------:|
| det_npu3 | Text Detection | 18.411 | **17.752** | +3.6% | 16.8 |
| cls_npu3 | Text Direction | 0.429 | **0.351** | +18.2% | 0.17 |
| rec_npu3 | Text Recognition | 2.104 | **1.719** | +18.3% | 1.4 |

### Analysis

PPOCR_v5 npu3 variants are significantly faster than npu1 (det: 18ms vs 29ms, rec: 1.7ms vs 3.7ms). The cls classifier at 0.35ms shows +18% — consistent with sub-millisecond pattern. The rec recognizer at 1.7ms also shows +18% — very strong for a ~2ms model. Native cls at 0.17ms means PCIe overhead (0.18ms) actually equals the computation time — this is the theoretical minimum achievable through PCIe Gen2 x1.

## Embedding — BGE-small-en-v1.5

### Test Configuration

- **Models**: [bge-small-en-v1.5](https://huggingface.co/AXERA-TECH/bge-small-en-v1.5) (69M, u16 npu3)
- **Tool**: `axcl_run_model`
- **Repeats**: 10 iterations, 3 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Native (ms) |
|-------|------------------:|--------------------:|--------:|------------:|
| bge-small-en-v1.5 | 35.414 | **34.859** | +1.6% | 32.4 |

### Analysis

At 35ms, the model is compute-bound with minimal PCIe overhead benefit (+1.6%). Optimized reaches 93% of native speed.

## VLM — Janus-Pro-1B (Component Benchmarks)

### Test Configuration

- **Models**: [Janus-Pro-1B](https://huggingface.co/AXERA-TECH/Janus-Pro-1B) — DeepSeek Janus Pro 1B, 24 LLaMA layers + post + SigLIP vision encoder
- **Tool**: `axcl_run_model`
- **Repeats**: 10 iterations, 3 warmup

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Vision (janus_warp_vit) | 143.761 | **142.996** | +0.5% |
| LLM Layer (llama l0) | 3.924 | **3.354** | +15.0% |
| LLM Post | 11.195 | **10.410** | +7.0% |

### Estimated Decode Speed

- Default: ~8 tok/s (24×3.924 + 11.195 ≈ 105.4 ms/tok)
- Optimized: ~11 tok/s (24×3.354 + 10.410 ≈ 90.9 ms/tok)

### Analysis

Janus-Pro-1B has larger LLM layers (~3.4ms) compared to InternVL (1ms), resulting in lower per-layer speedup (+15% vs +54-62%). However, the absolute improvement is still significant: ~8→11 tok/s. Vision encoder at 143ms is fully compute-bound. The LLaMA architecture in Janus differs from the Qwen2 used in InternVL, but the PCIe overhead pattern is consistent.

## VLM — InternVL3-1B (Component Benchmarks)

### Test Configuration

- **Models**: [InternVL3-1B](https://huggingface.co/AXERA-TECH/InternVL3-1B) — InternVL 3rd generation, 24 Qwen2 layers + post + InternViT vision encoder
- **Tool**: `axcl_run_model`
- **Repeats**: 10 iterations, 3 warmup

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Vision (internvl3_1b_vit) | 365.868 | **365.036** | +0.2% |
| LLM Layer (qwen2 l0) | 1.758 | **1.139** | +54.3% |
| LLM Post | 7.642 | **7.041** | +7.9% |

### Estimated Decode Speed

- Default: ~22 tok/s (24×1.758 + 7.642 ≈ 49.8 ms/tok)
- Optimized: ~29 tok/s (24×1.139 + 7.041 ≈ 34.4 ms/tok)

### Analysis

InternVL3-1B shows very similar performance to InternVL2.5-1B (both use 24 Qwen2 layers). Layer speedup +54% vs +62% — slightly less due to marginally larger layers (1.14ms vs 1.02ms). Both generations reach ~29-32 tok/s optimized. The vision encoder is slightly larger (365ms vs 358ms) but equally compute-bound.

## VLM — InternVL3.5-1B (Component Benchmarks)

### Test Configuration

- **Models**: [InternVL3_5-1B](https://huggingface.co/AXERA-TECH/InternVL3_5-1B) — InternVL 3.5 generation, 28 Qwen3 layers + post + InternViT vision encoder
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations (layer), 20 iterations (post), 10 iterations (vision), 3-5 warmup
- **Note**: Official native: 21.60 tok/s

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Vision (448x448) | 365.917 | **365.409** | +0.1% |
| LLM Layer (Qwen3 l0) | 1.915 | **1.526** | **+25.5%** |
| LLM Post | 8.512 | **8.134** | +4.7% |

### Estimated Decode Speed

- Default: ~19 tok/s (28×1.915 + 8.512 ≈ 62.1 ms/tok)
- Optimized: ~20 tok/s (28×1.526 + 8.134 ≈ 50.9 ms/tok)
- **Native: 21.60 tok/s** — optimization reaches 91% of native

### Analysis

InternVL3.5-1B uses Qwen3 architecture (vs Qwen2 in InternVL3-1B), with 28 layers instead of 24. The layer speedup (+25.5%) is lower than InternVL3-1B (+54%) because Qwen3 layers are larger (1.5ms vs 1.1ms). The vision encoder reuses InternViT (365ms, identical to InternVL3-1B). Optimized decode speed reaches **91% of native** (20 vs 21.6 tok/s).

## VLM — InternVL3.5-2B (Component Benchmarks)

### Test Configuration

- **Models**: [InternVL3_5-2B](https://huggingface.co/AXERA-TECH/InternVL3_5-2B) — InternVL 3.5 generation 2B, 28 Qwen3 layers + post + InternViT vision encoder
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations (layer), 10 iterations (post, vision), 3-5 warmup
- **Note**: Official native: 9.52 tok/s

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Vision (448x448) | 384.943 | **384.257** | +0.2% |
| LLM Layer (Qwen3 l0) | 3.609 | **3.318** | +8.8% |
| LLM Post | 16.360 | **15.671** | +4.4% |

### Estimated Decode Speed

- Default: ~9 tok/s (28×3.609 + 16.360 ≈ 117.4 ms/tok)
- Optimized: ~9.2 tok/s (28×3.318 + 15.671 ≈ 108.6 ms/tok)
- **Native: 9.52 tok/s** — optimization reaches 97% of native

### Analysis

InternVL3.5-2B has the same architecture as InternVL3.5-1B but with larger Qwen3 layers (3.3ms vs 1.5ms). The layer speedup (+8.8%) is lower than the 1B variant (+25.5%) — consistent with larger layers having less PCIe overhead. Vision encoder uses a slightly different InternViT (384ms vs 365ms). At 97% of native speed, the PCIe overhead is nearly eliminated.

## VLM — InternVL3-2B (Component Benchmarks)

### Test Configuration

- **Models**: [InternVL3-2B](https://huggingface.co/AXERA-TECH/InternVL3-2B) — InternVL 3rd generation 2B, 28 Qwen2 layers + post + InternViT vision encoder (W8A16)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations (layer), 10 iterations (post, vision), 3-5 warmup
- **Note**: Official native: 10 tok/s

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Vision (448x448) | 366.387 | **365.861** | +0.1% |
| LLM Layer (Qwen2 l0) | 3.192 | **3.039** | +5.0% |
| LLM Post | 12.057 | **11.699** | +3.1% |

### Estimated Decode Speed

- Default: ~10 tok/s (28×3.192 + 12.057 ≈ 101.4 ms/tok)
- Optimized: ~10.3 tok/s (28×3.039 + 11.699 ≈ 96.8 ms/tok)
- **Native: 10 tok/s** — optimization reaches **103% of native!**

### Analysis

InternVL3-2B (Qwen2 architecture) optimized speed slightly exceeds official native (10.3 vs 10 tok/s). This is likely because official benchmarks include Python runtime overhead. At 3ms per layer, the +5% optimization benefit is modest but sufficient to fully compensate for PCIe overhead. InternVL3-2B uses a "slim" ViT variant (366ms).

## VLM — SmolVLM2-500M-Video (Component Benchmarks)

### Test Configuration

- **Models**: [SmolVLM2-500M-Video-Instruct](https://huggingface.co/AXERA-TECH/SmolVLM2-500M-Video-Instruct_Ax650) — HuggingFace SmolVLM2 500M, 30 LLaMA layers + post + vision encoder (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations (layer), 50 iterations (post), 10 iterations (vision), 3-5 warmup
- **Note**: Official native: 35.23 tok/s (image), 35.32 tok/s (video)

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Vision (512x512) | 99.699 | **98.829** | +0.9% |
| LLM Layer (LLaMA l0) | 1.225 | **0.979** | **+25.1%** |
| LLM Post | 3.051 | **2.743** | +11.2% |

### Estimated Decode Speed

- Default: ~25 tok/s (30×1.225 + 3.051 ≈ 39.8 ms/tok)
- Optimized: ~31 tok/s (30×0.979 + 2.743 ≈ 32.1 ms/tok)
- **Native: 35.23 tok/s** — optimization reaches 88% of native

### Analysis

SmolVLM2-500M layers at 0.98ms show **+25% speedup** — consistent with sub-1ms models being very PCIe-latency-sensitive. The 30 LLaMA layers are slightly larger than SmolVLM2-256M layers. Post at 2.7ms also benefits significantly (+11%). At 31 tok/s via PCIe, this is the fastest 500M+ VLM measured. Native 35 tok/s means 88% efficiency.

## VLM — Qwen3-VL-2B (Component Benchmarks)

### Test Configuration

- **Models**: [Qwen3-VL-2B-Instruct](https://huggingface.co/AXERA-TECH/Qwen3-VL-2B-Instruct) — Alibaba Qwen3 vision-language, 28 layers + post + vision encoder (W8A16)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations (layer), 10 iterations (post, vision), 3-5 warmup
- **Note**: Official native: 9.5 tok/s, CMM 4.1 GiB

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Vision encoder | 158.200 | **157.564** | +0.4% |
| LLM Layer (Qwen3 l0) | 3.489 | **3.214** | +8.6% |
| LLM Post | 16.005 | **15.746** | +1.6% |

### Estimated Decode Speed

- Default: ~9 tok/s (28×3.489 + 16.005 ≈ 113.7 ms/tok)
- Optimized: ~9.5 tok/s (28×3.214 + 15.746 ≈ 105.7 ms/tok)
- **Native: 9.5 tok/s** — optimization **fully eliminates PCIe overhead!**

### Analysis

Qwen3-VL-2B is the first model where optimized PCIe speed **exactly matches** official native speed (9.5 tok/s). The LLM layers at 3.2ms are large enough to be compute-bound, with only 8.6% PCIe overhead per layer. The 28-layer architecture means the overhead is well-amortized. Vision encoder at 158ms is fully compute-bound.

## VLM — FastVLM-1.5B (Component Benchmarks)

### Test Configuration

- **Models**: [FastVLM-1.5B](https://huggingface.co/AXERA-TECH/FastVLM-1.5B) — Apple FastVLM 1.5B, 28 Qwen2 layers + post + image encoder (W8A16)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations (layer), 10 iterations (post, vision), 3-5 warmup
- **Note**: Official native: 11.53 tok/s, vision 512=58.56ms, 1024=231.07ms

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Vision (512x512) | 45.380 | **44.974** | +0.9% |
| LLM Layer (Qwen2 l0) | 3.175 | **2.739** | +15.9% |
| LLM Post | 12.292 | **11.528** | +6.6% |

### Estimated Decode Speed

- Default: ~10 tok/s (28×3.175 + 12.292 ≈ 101.2 ms/tok)
- Optimized: ~11.3 tok/s (28×2.739 + 11.528 ≈ 88.2 ms/tok)
- **Native: 11.53 tok/s** — optimization reaches **98% of native**

### Analysis

FastVLM-1.5B is the larger variant of FastVLM-0.5B (28 vs 24 layers, Qwen2 vs Qwen2). Layer speedup (+16%) is lower than FastVLM-0.5B (+32%) due to larger layers (2.7ms vs 1.2ms). Vision encoder (512x512) at 45ms shows minimal benefit. The optimized speed reaches 98% of native — practically no PCIe penalty. Note: Our C++ vision measurement (45ms) is faster than official Python (58.56ms) due to Python runtime overhead.

## VLM — MiniCPM-V-4 (Component Benchmarks)

### Test Configuration

- **Models**: [MiniCPM-V-4](https://huggingface.co/AXERA-TECH/MiniCPM-V-4) — OpenBMB MiniCPM-V 4th gen, 32 LLaMA layers + post + SigLIP vision encoder
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations (layer), 10 iterations (post, vision), 3-5 warmup

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| SigLIP vision | 581.429 | **580.968** | +0.1% |
| LLM Layer (LLaMA l0) | 5.914 | **5.280** | +12.0% |
| LLM Post | 9.531 | **9.241** | +3.1% |

### Estimated Decode Speed

- Default: ~5 tok/s (32×5.914 + 9.531 ≈ 198.8 ms/tok)
- Optimized: ~5.6 tok/s (32×5.280 + 9.241 ≈ 178.2 ms/tok)

### Analysis

MiniCPM-V-4 has the largest LLM layers tested (5.3ms, LLaMA architecture), resulting in moderate optimization benefit (+12%). The SigLIP vision encoder at 581ms is the heaviest VLM encoder measured — fully compute-bound. 32 layers make this a heavy model with ~178ms/tok decode time. MiniCPM-V-4 is a multimodal model supporting image, video, and document understanding.

## ASR — FireRedASR-AED

### Test Configuration

- **Models**: [FireRedASR-AED](https://huggingface.co/AXERA-TECH/FireRedASR-AED) — Xiaohongshu ASR AED-L model, encoder + decoder (Chinese/English, max 10s audio)
- **Tool**: `axcl_run_model`
- **Repeats**: 10 iterations, 3 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| Encoder (813M) | 162.220 | **161.491** | +0.5% |
| Decoder loop (397M) | 24.597 | **23.929** | +2.8% |

### Analysis

FireRedASR encoder at 161ms is compute-bound (+0.5%). The decoder at 24ms shows modest benefit (+2.8%). Both models are large (813M + 397M = 1.2GB total), using significant CMM (~1.3 GiB combined). This is a full-featured ASR model supporting Chinese and English with up to 10s audio input.

## Voice Activity Detection — SileroVAD

### Test Configuration

- **Models**: [SileroVAD](https://huggingface.co/AXERA-TECH/Spoken-Communication.axera) VAD component (1.1M model)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup | Optimized FPS |
|-------|------------------:|--------------------:|--------:|--------------:|
| SileroVAD | 6.332 | **5.821** | +8.8% | 172 |

### Analysis

SileroVAD at 5.8ms shows +8.8% optimization benefit — consistent with models in the 5-7ms latency range. At 172 inferences/sec, it can process audio frames much faster than real-time. This is a critical component in voice pipelines for detecting speech segments before sending to ASR.

## CLIP — cnclip ViT-L/14-336px (Chinese CLIP)

### Test Configuration

- **Models**: [cnclip](https://huggingface.co/AXERA-TECH/cnclip) — Chinese CLIP ViT-L/14-336px, text and vision encoders (u16)
- **Tool**: `axcl_run_model`
- **Repeats**: 10 iterations, 3 warmup

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Text (u16) | 5.044 | **4.351** | +14.0% |
| Vision (u16) | 114.130 | **113.359** | +0.7% |

### Analysis

cnclip text encoder at 4.4ms shows +14% — consistent with other text encoders in this latency range (LibCLIP cnclip text: 4.6ms +10%, CLIP ViT-L/14 text: 5.8ms +10%). Vision encoder at 113ms is compute-bound (+0.7%).

## Portrait Animation — LivePortrait

### Test Configuration

- **Models**: [LivePortrait](https://huggingface.co/AXERA-TECH/LivePortrait) — 4 components (feature extractor, motion extractor, spade generator, stitching/retargeting) (pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| stitching_retargeting | 0.311 | **0.198** | **+57.1%** |
| motion_extractor | 8.177 | **7.472** | +9.4% |
| feature_extractor | 20.364 | **19.866** | +2.5% |
| spade_generator | 233.282 | **232.544** | +0.3% |

### Analysis

LivePortrait components span a 1000x range of inference times (0.2ms to 233ms), providing an excellent demonstration of the optimization pattern:
- **stitching** (0.2ms): **+57%** — the tiny stitching network is extremely PCIe-latency-sensitive
- **motion** (7.5ms): +9% — moderate benefit
- **feature** (20ms): +3% — approaching compute-bound territory
- **spade** (233ms): +0.3% — fully compute-bound, the heavy generator dominates

The full LivePortrait pipeline (feature + motion + spade + stitching) takes ~260ms per frame (~4 FPS) via PCIe. The spade generator dominates total time.

## ASR — Zipformer

### Test Configuration

- **Models**: [Zipformer.axera](https://huggingface.co/AXERA-TECH/Zipformer.axera) — encoder, decoder, joiner (Kaldi/icefall architecture, pre-compiled for AX650)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| joiner | 0.664 | **0.344** | **+93.0%** |
| decoder | 0.437 | **0.243** | **+79.8%** |
| encoder | 3.602 | **3.018** | +19.4% |

### Analysis

Zipformer joiner at **+93%** is the **absolute record speedup** measured across all 120+ models — beating the previous record of +71% (OCR classifier). The decoder at +80% is the second highest.

Both joiner and decoder are ultra-fast sub-millisecond models where PCIe round-trip latency (~0.3ms) is a massive fraction of total time. Moving IRQ from slow A55 to fast A76 nearly halves the overhead.

The encoder at 3ms shows +19% — still a strong benefit, consistent with other models in this latency range.

For streaming ASR, the joiner and decoder are called once per frame (typically every 60ms), so the absolute time savings (~0.3ms + ~0.2ms per frame) are meaningful for low-latency applications.

## Speaker Embedding — CAM++ (3D-Speaker)

### Test Configuration

- **Models**: [3D-Speaker-MT.Axera](https://huggingface.co/AXERA-TECH/3D-Speaker-MT.Axera) CAM++ speaker embedding (11M, for speaker identification/diarization)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup
- **Note**: Part of 3D-Speaker pipeline (VAD + CAM++ + SenseVoice)

### With vs Without Optimization

| Model | Default avg (ms) | Optimized avg (ms) | Speedup |
|-------|------------------:|--------------------:|--------:|
| CAM++ (campplus) | 3.634 | **3.107** | +17.0% |

### Analysis

CAM++ at 3.1ms shows +17% optimization benefit — consistent with models in the 3-4ms latency range. This is a speaker embedding model used for speaker diarization in meeting transcription. At 322 inferences/sec, it can process audio segments much faster than real-time. The existing 3D-Speaker ECAPA-TDNN (3.9ms, +3%) uses a different architecture; CAM++ is significantly more PCIe-latency-sensitive despite similar inference time.

## LLM — Qwen3-0.6B GPTQ-Int4 (Component Benchmarks)

### Test Configuration

- **Models**: [Qwen3-0.6B-GPTQ-Int4](https://huggingface.co/AXERA-TECH/Qwen3-0.6B-GPTQ-Int4) — Alibaba Qwen3 0.6B, 28 layers + post (W4A16), C256 P1024 CTX2047
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations (layer), 20 iterations (post), 3-5 warmup
- **Note**: Official native (W8A16): 19-20 tok/s. End-to-end W8A16 optimized: 10-12 tok/s.

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| LLM Layer (Qwen3 l0) | 1.676 | **1.357** | +23.5% |
| LLM Post (CMM 170M) | 8.655 | **8.125** | +6.5% |

### Estimated Decode Speed

- Default: ~18 tok/s (28×1.676 + 8.655 ≈ 55.6 ms/tok)
- Optimized: ~21.7 tok/s (28×1.357 + 8.125 ≈ 46.1 ms/tok)
- **Native (W8A16): 19-20 tok/s** — Int4 optimization **exceeds native W8A16!**

### Analysis

Qwen3-0.6B GPTQ-Int4 estimated 21.7 tok/s exceeds official native W8A16 speed (19-20 tok/s). The W8A16 end-to-end optimized benchmark showed 10-12 tok/s because it includes tokenization, prompt processing, and Python/C++ runtime overhead. The component benchmark isolates pure NPU inference, showing that Int4 quantization makes the 0.6B model fast enough to overcome PCIe overhead. Layer speedup (+23.5%) at 1.4ms is consistent with the InternVL3.5-1B Int4 result (+26% at 1.2ms). This confirms a general pattern: **Int4 quantization + PCIe optimization can exceed native W8A16 performance** on the AX650N.

## Translation LLM — HY-MT1.5-1.8B (Component Benchmarks)

### Test Configuration

- **Models**: [HY-MT1.5-1.8B_GPTQ_INT4](https://huggingface.co/AXERA-TECH/HY-MT1.5-1.8B_GPTQ_INT4) — Tencent HuanYuan MT 1.5 translation model, 32 layers + post (W4A16)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations (layer), 10 iterations (post), 3-5 warmup
- **Note**: Supports 38 languages including zh/en/ja/ko/ru/fr/de/es. Context: 2K, prefill: 1K.

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| LLM Layer (HuanYuan l0) | 2.045 | **1.826** | +12.0% |
| LLM Post (CMM 269M) | 12.891 | **12.415** | +3.8% |

### Estimated Decode Speed

- Default: ~12.8 tok/s (32×2.045 + 12.891 ≈ 78.3 ms/tok)
- Optimized: ~14.1 tok/s (32×1.826 + 12.415 ≈ 70.8 ms/tok)

### Analysis

HY-MT1.5-1.8B is a dedicated translation LLM with 32 HuanYuan layers at W4A16 quantization. At 14.1 tok/s optimized, it's surprisingly fast for a 1.8B model — faster than Qwen3-0.6B (10-12 tok/s) despite being 3x larger! The INT4 quantization and translation-optimized architecture (HuanYuan Dense) explains the efficiency. Layer speedup (+12%) is consistent with the 1.8ms inference time. This is the first non-Qwen/non-LLaMA LLM architecture benchmarked.

## LLM — Qwen3-4B-Instruct-2507 GPTQ-Int4 (Component Benchmarks)

### Test Configuration

- **Models**: [Qwen3-4B-Instruct-2507-GPTQ-Int4](https://huggingface.co/AXERA-TECH/3D-Speaker-Meeting-Summary) — Qwen3 4B Instruct (July 2025 version), 36 layers + post (W4A16), 8K context
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations (layer), 10 iterations (post), 3-5 warmup
- **Note**: Part of 3D-Speaker-Meeting-Summary pipeline. CMM: 424M (post), p256 prefill.

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| LLM Layer (Qwen3 l0) | 7.631 | **7.301** | +4.5% |
| LLM Post (CMM 424M) | 19.270 | **18.836** | +2.3% |

### Estimated Decode Speed

- Default: ~3.4 tok/s (36×7.631 + 19.270 ≈ 294.0 ms/tok)
- Optimized: ~3.55 tok/s (36×7.301 + 18.836 ≈ 281.7 ms/tok)

### Analysis

Qwen3-4B-2507 Int4 shows similar performance to the existing Qwen3-4B W8A16 (3.7 tok/s). The Int4 quantization doesn't provide a speed advantage because the 4B model is compute-bound at 7.3ms/layer — PCIe overhead is already minimal. The 8K context and p256 prefill configuration adds overhead compared to the p128 W8A16 version. Layer speedup (+4.5%) confirms that models at 7ms+ are firmly in compute-bound territory.

## VLM — Qwen2.5-VL-3B-Instruct GPTQ-Int4 (Component Benchmarks)

### Test Configuration

- **Models**: [Qwen2.5-VL-3B-Instruct-GPTQ-Int4](https://huggingface.co/AXERA-TECH/Qwen2.5-VL-3B-Instruct-GPTQ-Int4) — Alibaba Qwen2.5 vision-language 3B, 36 layers + post + vision encoder (W4A16)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations (layer), 10 iterations (post, vision), 3-5 warmup
- **Note**: CMM: 340M (post), 807M (vision). Vision input: 392x392.

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| Vision encoder (392x392) | 560.840 | **559.902** | +0.2% |
| LLM Layer (Qwen2.5 l0) | 2.948 | **2.425** | +21.5% |
| LLM Post (CMM 340M) | 16.147 | **15.452** | +4.5% |

### Estimated Decode Speed

- Default: ~8.2 tok/s (36×2.948 + 16.147 ≈ 122.3 ms/tok)
- Optimized: ~9.7 tok/s (36×2.425 + 15.452 ≈ 102.8 ms/tok)

### Analysis

Qwen2.5-VL-3B is the first Qwen2.5-VL generation model benchmarked. The vision encoder at 560ms is the second heaviest measured (after MiniCPM-V-4 SigLIP at 581ms), using 807M CMM — fully compute-bound. Layer speedup (+21.5%) at 2.4ms is excellent, giving an estimated 9.7 tok/s — competitive with Qwen3-VL-2B (9.5 tok/s). The 36-layer Qwen2.5 architecture with Int4 quantization achieves good efficiency. This is the largest VLM (3B) that still delivers near-10 tok/s decode speed on the AX650N via PCIe.

## VLM — InternVL3.5-1B GPTQ-Int4 (Component Benchmarks)

### Test Configuration

- **Models**: [InternVL3_5-1B_GPTQ_INT4](https://huggingface.co/AXERA-TECH/InternVL3_5-1B_GPTQ_INT4) — InternVL 3.5 1B, 28 Qwen3 layers + post (W4A16)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations (layer), 20 iterations (post), 3-5 warmup
- **Note**: Vision encoder not benchmarked (same InternViT as W8A16). Official native (W8A16): 21.60 tok/s.

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| LLM Layer (Qwen3 l0) | 1.457 | **1.156** | +26.0% |
| LLM Post (CMM 170M) | 8.334 | **8.002** | +4.1% |

### Estimated Decode Speed

- Default: ~20.4 tok/s (28×1.457 + 8.334 ≈ 49.1 ms/tok)
- Optimized: ~24.8 tok/s (28×1.156 + 8.002 ≈ 40.4 ms/tok)
- **Native (W8A16): 21.60 tok/s** — Int4 optimization **exceeds native W8A16 by 15%!**

### W4A16 vs W8A16 Comparison (InternVL3.5-1B)

| Metric | W8A16 | W4A16 (Int4) | Int4 advantage |
|--------|------:|-------------:|---------------:|
| Layer (optimized) | 1.526 ms | **1.156 ms** | -24% faster |
| Post (optimized) | 8.134 ms | **8.002 ms** | -2% faster |
| Estimated tok/s (opt) | ~20 | **~24.8** | +24% faster |

### Analysis

InternVL3.5-1B GPTQ-Int4 layers are **24% faster** than W8A16 at identical architecture — Int4 quantization reduces data transfer and computation per layer. At 1.16ms/layer, the optimization speedup (+26%) is consistent with the W8A16 variant (+25.5%). The estimated 24.8 tok/s **exceeds official native W8A16 speed** (21.6 tok/s) by 15%, demonstrating that Int4 quantization can compensate for PCIe overhead and even outperform the un-quantized native baseline.

## TTS — VoxCPM (Component Benchmarks)

### Test Configuration

- **Models**: [VoxCPM](https://huggingface.co/AXERA-TECH/VoxCPM) — OpenBMB VoxCPM TTS, multi-component architecture using MiniCPM layers + DiT + FSQ (W8A16)
- **Tool**: `axcl_run_model`
- **Repeats**: 100 iterations, 5 warmup (10 iterations for post)
- **Architecture**: base_lm (24 MiniCPM layers + post), feat_encoder (4 MiniCPM layers), residual_lm (6 layers), feat_decoder (4 layers), LocDiT (3 parts), FSQ, stop_predictor

### With vs Without Optimization

| Component | Default avg (ms) | Optimized avg (ms) | Speedup |
|-----------|------------------:|--------------------:|--------:|
| stop_predictor | 0.552 | **0.331** | **+66.8%** |
| FSQ layer | 0.470 | **0.341** | **+37.8%** |
| LocDiT part1 | 0.693 | **0.453** | **+53.0%** |
| base_lm LLM layer (MiniCPM l0, 24 layers) | 1.559 | **0.995** | **+56.7%** |
| feat_encoder LLM layer (MiniCPM l0, 4 layers) | 1.510 | **1.051** | **+43.7%** |
| base_lm LLM post (CMM 82M) | 4.273 | **4.185** | +2.1% |

### Estimated base_lm Decode Speed

- Default: ~24 tok/s (24×1.559 + 4.273 ≈ 41.7 ms/tok)
- Optimized: ~36 tok/s (24×0.995 + 4.185 ≈ 28.1 ms/tok)

### Analysis

VoxCPM is a revelation for optimization impact on TTS. The stop_predictor at **+66.8%** is the **third highest speedup** ever measured (after Zipformer joiner +93% and decoder +80%). All sub-millisecond components show massive gains:
- **stop_predictor** (0.33ms): +67% — ultra-fast binary predictor, PCIe latency dominant
- **FSQ** (0.34ms): +38% — finite scalar quantizer, tiny compute
- **LocDiT part1** (0.45ms): +53% — diffusion transformer block
- **base_lm layer** (1.0ms): +57% — MiniCPM layers at 1ms are in the sweet spot

The base_lm (24 MiniCPM layers) alone would decode at 36 tok/s optimized vs 24 tok/s default — a **50% speedup** for the main LLM backbone. VoxCPM's TTS quality depends heavily on iterative processing through multiple sub-networks, making PCIe optimization critical for end-to-end latency.

## VLM — Xiaomi-MiMo-VL-Miloco-7B GPTQ-Int4 (Component Benchmarks)

### Test Configuration

- **Model**: [AXERA-TECH/Xiaomi-MiMo-VL-Miloco-7B-AX650-GPTQ-Int4](https://huggingface.co/AXERA-TECH/Xiaomi-MiMo-VL-Miloco-7B-AX650-GPTQ-Int4)
- **Architecture**: Xiaomi MiMo-VL 7B (Qwen2.5-VL backbone), W4A16 GPTQ quantization
- **Components**: 36 decoder layers (layer0 113MB), post (648MB), vision encoder
- **NPU**: 3-core mode
- **Benchmark**: axcl_run_model, warmup 5, repeat 30

### With vs Without Optimization

| Component | Optimized (ms) | Default (ms) | Improvement |
|-----------|:-:|:-:|:-:|
| LLM Layer (×36) | 5.020 | 5.715 | **+13.8%** |
| LLM Post | 30.064 | 30.487 | **+1.4%** |

### Estimated Decode Speed

- **Optimized**: 36 × 5.020 + 30.064 = 210.8ms → **4.7 tok/s**
- **Default**: 36 × 5.715 + 30.487 = 236.2ms → **4.2 tok/s**
- **Optimization gain**: +12.1%

### Analysis

Xiaomi-MiMo-VL-7B is the first non-Qwen/non-InternVL vision-language model tested, and the largest VLM benchmarked. At 4.7 tok/s, it's usable for visual question answering. The 7B model with 36 layers uses Qwen2.5-VL architecture under the hood. Layer optimization at +13.8% is moderate for a 5ms model, and the massive 648MB post (30ms) shows minimal optimization effect as expected for large compute-bound operations.

## VLM — Qwen3-VL-2B-Instruct GPTQ-Int4 (Component Benchmarks)

### Test Configuration

- **Model**: [AXERA-TECH/Qwen3-VL-2B-Instruct-GPTQ-Int4](https://huggingface.co/AXERA-TECH/Qwen3-VL-2B-Instruct-GPTQ-Int4)
- **Architecture**: Qwen3 VL 2B, W4A16 GPTQ quantization
- **Components**: 28 decoder layers (layer0 38MB), post (324MB)
- **NPU**: 3-core mode
- **Benchmark**: axcl_run_model, warmup 5, repeat 30

### With vs Without Optimization

| Component | Optimized (ms) | Default (ms) | Improvement |
|-----------|:-:|:-:|:-:|
| LLM Layer (×28) | 2.115 | 2.521 | **+19.2%** |
| LLM Post | 15.646 | 15.937 | **+1.9%** |

### Estimated Decode Speed

- **Optimized**: 28 × 2.115 + 15.646 = 74.9ms → **13.4 tok/s**
- **Default**: 28 × 2.521 + 15.937 = 86.5ms → **11.6 tok/s**
- **Optimization gain**: +15.5%

### W4A16 vs W8A16 Comparison (Qwen3-VL-2B)

| Metric | W4A16 (Int4) | W8A16 |
|--------|:-:|:-:|
| Layer time (opt) | 2.115ms | ~3.2ms |
| Decode speed (opt) | 13.4 tok/s | ~9.5 tok/s |
| Int4 advantage | **+41%** | — |

### Analysis

Qwen3-VL-2B Int4 delivers 13.4 tok/s — 41% faster than W8A16 (~9.5 tok/s estimated). The +19.2% layer speedup is consistent with the ~2ms optimization pattern. Multiple context/prefill variants exist on HuggingFace (C256/C512/P1536/P3584), but all share the same decoder layer architecture — only vision resolution and context length differ.

## LLM — Qwen2.5-0.5B-Instruct GPTQ-Int4 (Component Benchmarks)

### Test Configuration

- **Model**: [AXERA-TECH/Qwen2.5-0.5B-Instruct-GPTQ-Int4](https://huggingface.co/AXERA-TECH/Qwen2.5-0.5B-Instruct-GPTQ-Int4)
- **Architecture**: Qwen2.5 0.5B, W4A16 GPTQ quantization
- **Components**: 24 decoder layers (layer0 12MB), post (142MB)
- **NPU**: 3-core mode
- **Benchmark**: axcl_run_model, warmup 5, repeat 30

### With vs Without Optimization

| Component | Optimized (ms) | Default (ms) | Improvement |
|-----------|:-:|:-:|:-:|
| LLM Layer (×24) | 0.889 | 1.185 | **+33.3%** |
| LLM Post | 6.964 | 7.399 | **+6.2%** |

### Estimated Decode Speed

- **Optimized**: 24 × 0.889 + 6.964 = 28.3ms → **35.3 tok/s**
- **Default**: 24 × 1.185 + 7.399 = 35.8ms → **27.9 tok/s**
- **Optimization gain**: +26.5%

### Analysis

Qwen2.5-0.5B Int4 at **35.3 tok/s** is the fastest LLM decode speed measured on AX650N via PCIe. The sub-millisecond layer (0.889ms) puts it in the same territory as VoxCPM base_lm (1.0ms, +57%) and SmolVLM-256M (0.45ms, +56%), where PCIe latency dominates and optimization has maximum impact. The +33.3% layer speedup confirms this — among the highest for any LLM layer tested. This model is ideal for real-time chatbot applications on edge devices.

## VLM — FastVLM-1.5B GPTQ-Int4 (Component Benchmarks)

### Test Configuration

- **Model**: [AXERA-TECH/FastVLM-1.5B-GPTQ-Int4](https://huggingface.co/AXERA-TECH/FastVLM-1.5B-GPTQ-Int4)
- **Architecture**: FastVLM 1.5B (LLaVA-Qwen2), W4A16 GPTQ quantization, context 1K, prefill 640
- **Components**: 28 decoder layers (layer0 29MB), post (243MB)
- **NPU**: 3-core mode
- **Benchmark**: axcl_run_model, warmup 5, repeat 30

### With vs Without Optimization

| Component | Optimized (ms) | Default (ms) | Improvement |
|-----------|:-:|:-:|:-:|
| LLM Layer (×28) | 1.461 | 1.470 | **+0.6%** |
| LLM Post | 11.659 | 12.199 | **+4.6%** |

### Estimated Decode Speed

- **Optimized**: 28 × 1.461 + 11.659 = 52.6ms → **19.0 tok/s**
- **Default**: 28 × 1.470 + 12.199 = 53.4ms → **18.7 tok/s**
- **Optimization gain**: +1.5%

### W4A16 vs W8A16 Comparison (FastVLM-1.5B)

| Metric | W4A16 (Int4) | W8A16 |
|--------|:-:|:-:|
| Layer time (opt) | 1.461ms | 1.241ms |
| Post time (opt) | 11.659ms | 12.117ms |
| Decode speed (opt) | 19.0 tok/s | 17.2 tok/s |

### Analysis

Anomalous result: FastVLM-1.5B Int4 layer shows essentially **zero optimization benefit** (+0.6%). The W8A16 layer (1.241ms) was actually faster than Int4 (1.461ms), which is unexpected — Int4 should have smaller layers. This suggests the Int4 quantization introduces computational overhead (dequantization) that exceeds the data transfer savings, and the layer is purely compute-bound with no PCIe bottleneck. The 19.0 tok/s is still 10% faster than W8A16 overall due to post model savings.

## VLM — InternVL3.5-2B GPTQ-Int4 (Component Benchmarks)

### Test Configuration

- **Model**: [AXERA-TECH/InternVL3_5-2B_GPTQ_INT4](https://huggingface.co/AXERA-TECH/InternVL3_5-2B_GPTQ_INT4)
- **Architecture**: InternVL3.5 2B (Qwen3 LLM backbone), W4A16 GPTQ quantization
- **Components**: 28 decoder layers (layer0 34MB), post (325MB)
- **NPU**: 3-core mode
- **Benchmark**: axcl_run_model, warmup 5, repeat 30

### With vs Without Optimization

| Component | Optimized (ms) | Default (ms) | Improvement |
|-----------|:-:|:-:|:-:|
| LLM Layer (×28) | 1.897 | 2.400 | **+26.5%** |
| LLM Post | 15.585 | 16.305 | **+4.6%** |

### Estimated Decode Speed

- **Optimized**: 28 × 1.897 + 15.585 = 68.7ms → **14.6 tok/s**
- **Default**: 28 × 2.400 + 16.305 = 83.5ms → **12.0 tok/s**
- **Optimization gain**: +21.2%

### W4A16 vs W8A16 Comparison (InternVL3.5-2B)

| Metric | W4A16 (Int4) | W8A16 |
|--------|:-:|:-:|
| Layer time (opt) | 1.897ms | 3.186ms |
| Post time (opt) | 15.585ms | 15.688ms |
| Decode speed (opt) | 14.6 tok/s | 8.1 tok/s |
| Int4 advantage | **+80%** | — |

### Analysis

InternVL3.5-2B Int4 delivers 14.6 tok/s — **80% faster** than W8A16 (8.1 tok/s). This is the largest W4/W8 gap measured so far, because Int4 nearly halves the layer time (1.9ms vs 3.2ms) while post remains identical. The +26.5% layer optimization is strong and consistent with the ~2ms pattern. For a 2B VLM, 14.6 tok/s is excellent for real-time visual question answering.

## LLM — DeepSeek-R1-Distill-Qwen-1.5B GPTQ-Int4 (Component Benchmarks)

### Test Configuration

- **Model**: [AXERA-TECH/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ-Int4](https://huggingface.co/AXERA-TECH/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ-Int4)
- **Architecture**: Qwen2 1.5B (DeepSeek-R1 distilled), W4A16 GPTQ quantization
- **Components**: 28 decoder layers (layer0 27MB), post (243MB)
- **NPU**: 3-core mode
- **Benchmark**: axcl_run_model, warmup 5, repeat 30

### With vs Without Optimization

| Component | Optimized (ms) | Default (ms) | Improvement |
|-----------|:-:|:-:|:-:|
| LLM Layer (×28) | 1.592 | 1.980 | **+24.4%** |
| LLM Post | 11.611 | 12.194 | **+5.0%** |

### Estimated Decode Speed

- **Optimized**: 28 × 1.592 + 11.611 = 56.2ms → **17.8 tok/s**
- **Default**: 28 × 1.980 + 12.194 = 67.6ms → **14.8 tok/s**
- **Optimization gain**: +20.3%

### W4A16 vs W8A16 Comparison (DeepSeek-R1-1.5B)

| Metric | W4A16 (GPTQ-Int4) | W8A16 (end-to-end) | W4A16 (end-to-end) |
|--------|:-:|:-:|:-:|
| Decode speed (opt) | **17.8 tok/s** (est.) | 10.2-11.0 tok/s | 7.6-8.6 tok/s |
| vs W8A16 e2e | **+62%** | — | — |

### Analysis

DeepSeek-R1-1.5B Int4 at 17.8 tok/s estimated is remarkably fast — 62% faster than W8A16 end-to-end (10.2-11.0 tok/s). The previous W4A16 end-to-end result (7.6-8.6 tok/s) was measured with the older `main_axcl_aarch64` binary which includes tokenizer overhead, while this component benchmark measures pure NPU time. The +24.4% layer speedup is among the highest for LLM layers, consistent with the sub-2ms optimization pattern.

## LLM — Qwen3-1.7B GPTQ-Int4 (Component Benchmarks)

### Test Configuration

- **Model**: [AXERA-TECH/Qwen3-1.7B-GPTQ-Int4](https://huggingface.co/AXERA-TECH/Qwen3-1.7B-GPTQ-Int4)
- **Architecture**: Qwen3 1.7B, W4A16 GPTQ quantization, context 256 prefill 3584
- **Components**: 28 decoder layers (layer0 43MB), post (325MB)
- **NPU**: 3-core mode
- **Benchmark**: axcl_run_model, warmup 5, repeat 30

### With vs Without Optimization

| Component | Optimized (ms) | Default (ms) | Improvement |
|-----------|:-:|:-:|:-:|
| LLM Layer (×28) | 2.557 | 2.835 | **+10.9%** |
| LLM Post | 15.688 | 16.179 | **+3.1%** |

### Estimated Decode Speed

- **Optimized**: 28 × 2.557 + 15.688 = 87.3ms → **11.5 tok/s**
- **Default**: 28 × 2.835 + 16.179 = 95.6ms → **10.5 tok/s**
- **Optimization gain**: +9.5%

### W4A16 vs W8A16 Comparison (Qwen3-1.7B)

| Metric | W4A16 (Int4) | W8A16 (measured) |
|--------|:-:|:-:|
| Layer time (opt) | 2.557ms | ~3.5ms (estimated from MEMORY) |
| Decode speed (opt) | 11.5 tok/s | 7.8-8.0 tok/s |
| Improvement | **+44% vs W8A16** | — |

### Analysis

Qwen3-1.7B Int4 at 11.5 tok/s is a significant upgrade over W8A16 (7.8-8.0 tok/s from previous benchmarks), delivering a 44% speed increase through quantization alone. The +10.9% layer optimization is moderate for a 2.5ms model, and the c256 prefix context (vs c128 for most others) doesn't noticeably impact single-token decode speed.

## LLM — Qwen2.5-3B-Instruct (Component Benchmarks)

### Test Configuration

- **Model**: [AXERA-TECH/Qwen2.5-3B-Instruct](https://huggingface.co/AXERA-TECH/Qwen2.5-3B-Instruct)
- **Architecture**: Qwen2.5 3B, both W8A16 and W4A16 (GPTQ-Int4) variants
- **Components**: 36 decoder layers + post (324MB shared)
- **NPU**: 3-core mode
- **Benchmark**: axcl_run_model, warmup 5, repeat 30

### With vs Without Optimization

| Variant | Component | Optimized (ms) | Default (ms) | Improvement |
|---------|-----------|:-:|:-:|:-:|
| W8A16 | LLM Layer (×36) | 3.861 | 4.436 | **+14.9%** |
| W8A16 | LLM Post | 15.803 | 16.116 | **+2.0%** |
| W4A16 | LLM Layer (×36) | 2.506 | 2.678 | **+6.9%** |
| W4A16 | LLM Post | 15.796 | 16.383 | **+3.7%** |

### Estimated Decode Speed

| Variant | Optimized | Default | Gain |
|---------|:-:|:-:|:-:|
| W8A16 | 36×3.861 + 15.803 = 154.8ms → **6.5 tok/s** | 175.8ms → **5.7 tok/s** | +13.6% |
| W4A16 | 36×2.506 + 15.796 = 106.0ms → **9.4 tok/s** | 112.8ms → **8.9 tok/s** | +6.4% |

### W4A16 vs W8A16 Comparison

| Metric | W4A16 (Int4) | W8A16 | Int4 advantage |
|--------|:-:|:-:|:-:|
| Layer time (opt) | 2.506ms | 3.861ms | 35% faster |
| Decode speed (opt) | 9.4 tok/s | 6.5 tok/s | +45% |
| Decode speed (default) | 8.9 tok/s | 5.7 tok/s | +56% |

### Analysis

Qwen2.5-3B at 9.4 tok/s (Int4, optimized) is faster than Qwen3-4B W8A16 (3.7 tok/s) despite being a similarly-sized model, showing the massive impact of Int4 quantization. The W8A16 variant shows a consistent +14.9% layer speedup, while Int4 layers show only +6.9% — the smaller Int4 layers (2.5ms) are already quite fast, and the PCIe overhead is less dominant at this layer size. Post model timings are identical between variants, confirming they share the same embedding layer.

## LLM — Qwen2.5-1.5B-Instruct (Component Benchmarks)

### Test Configuration

- **Model**: [AXERA-TECH/Qwen2.5-1.5B-Instruct](https://huggingface.co/AXERA-TECH/Qwen2.5-1.5B-Instruct)
- **Architecture**: Qwen2.5 1.5B, both W8A16 and W4A16 (GPTQ-Int4) variants
- **Components**: 28 decoder layers + post (243MB shared)
- **NPU**: 3-core mode
- **Benchmark**: axcl_run_model, warmup 5, repeat 30

### With vs Without Optimization

| Variant | Component | Optimized (ms) | Default (ms) | Improvement |
|---------|-----------|:-:|:-:|:-:|
| W8A16 | LLM Layer (×28) | 2.553 | 2.933 | **+14.9%** |
| W8A16 | LLM Post | 11.704 | 12.222 | **+4.4%** |
| W4A16 | LLM Layer (×28) | 1.722 | 2.209 | **+28.3%** |
| W4A16 | LLM Post | 11.733 | 11.899 | **+1.4%** |

### Estimated Decode Speed

| Variant | Optimized | Default | Gain |
|---------|:-:|:-:|:-:|
| W8A16 | 28×2.553 + 11.704 = 83.2ms → **12.0 tok/s** | 94.3ms → **10.6 tok/s** | +13.4% |
| W4A16 | 28×1.722 + 11.733 = 59.9ms → **16.7 tok/s** | 73.8ms → **13.6 tok/s** | +23.0% |

### W4A16 vs W8A16 Comparison

| Metric | W4A16 (Int4) | W8A16 | Int4 advantage |
|--------|:-:|:-:|:-:|
| Layer time (opt) | 1.722ms | 2.553ms | 33% faster |
| Decode speed (opt) | 16.7 tok/s | 12.0 tok/s | +39% |
| Decode speed (default) | 13.6 tok/s | 10.6 tok/s | +28% |

### Analysis

Qwen2.5-1.5B provides a clean W4A16 vs W8A16 comparison at the 1.5B scale. Int4 quantization delivers a massive 39% speed advantage (16.7 vs 12.0 tok/s) with optimization, and 28% without. The Int4 layer at 1.722ms shows a huge +28.3% optimization gain — among the highest for LLM layers, confirming that smaller/faster layers benefit more from PCIe optimization. Post model is identical between variants (same 243MB, same timing), as expected since it only contains the embedding layer.

## LLM — Gemma-3-1B-it (Component Benchmarks)

### Test Configuration

- **Model**: [AXERA-TECH/Gemma-3-1B-it](https://huggingface.co/AXERA-TECH/Gemma-3-1B-it)
- **Architecture**: Gemma 3 1B (Google), W8A16 quantization
- **Components**: 26 decoder layers (layer0 47MB), post (321MB)
- **NPU**: 3-core mode
- **Benchmark**: axcl_run_model, warmup 5, repeat 30

### With vs Without Optimization

| Component | Optimized (ms) | Default (ms) | Improvement |
|-----------|:-:|:-:|:-:|
| LLM Layer (×26) | 1.970 | 2.419 | **+22.8%** |
| LLM Post | 15.458 | 16.227 | **+4.7%** |

### Estimated Decode Speed

- **Optimized**: 26 × 1.970 + 15.458 = 66.7ms → **15.0 tok/s**
- **Default**: 26 × 2.419 + 16.227 = 79.1ms → **12.6 tok/s**
- **Optimization gain**: +18.6%

### Analysis

Gemma-3-1B is the first Google architecture tested on AX650N. At 15.0 tok/s optimized, it's competitive with InternVL3.5-1B Int4 (24.8 tok/s) when accounting for the difference in quantization: Gemma-3 is W8A16 while InternVL3.5-1B Int4 is W4A16 (half the data per layer). The +22.8% layer speedup is high for a ~2ms model, consistent with the optimization pattern. Initial 10-repeat benchmarks showed misleading results due to CPU cache warmth from the previous run — 30 repeats gave reliable data.

## LLM — DeepSeek-R1-Distill-Qwen-7B GPTQ-Int4 (Component Benchmarks)

### Test Configuration

- **Model**: [AXERA-TECH/DeepSeek-R1-Distill-Qwen-7B-GPTQ-Int4](https://huggingface.co/AXERA-TECH/DeepSeek-R1-Distill-Qwen-7B-GPTQ-Int4)
- **Architecture**: Qwen2 7B (DeepSeek-R1 distilled), W4A16 GPTQ quantization
- **Components**: 28 decoder layers (layer0 130MB), post (567MB)
- **NPU**: 3-core mode
- **Benchmark**: axcl_run_model, warmup 3, repeat 10

### With vs Without Optimization

| Component | Optimized (ms) | Default (ms) | Improvement |
|-----------|:-:|:-:|:-:|
| LLM Layer (×28) | 6.319 | 6.945 | **+9.9%** |
| LLM Post | 26.251 | 26.903 | **+2.5%** |

### Estimated Decode Speed

- **Optimized**: 28 × 6.319 + 26.251 = 203.2ms → **4.9 tok/s**
- **Default**: 28 × 6.945 + 26.903 = 221.4ms → **4.5 tok/s**
- **Optimization gain**: +8.9%

### Analysis

DeepSeek-R1-7B Int4 at 4.9 tok/s outperforms Qwen2.5-7B W4A16 (4.4 tok/s) by 11%, likely due to the newer GPTQ-Int4 quantization format vs the older W4A16. Both are 7B Qwen2-based models, so the architecture is identical — the difference is purely quantization efficiency. For a reasoning model, 4.9 tok/s is usable, though the long `<think>` blocks typical of R1 will add latency.

## VLM — Qwen3-VL-4B-Instruct GPTQ-Int4 (Component Benchmarks)

### Test Configuration

- **Model**: [AXERA-TECH/Qwen3-VL-4B-Instruct-GPTQ-Int4](https://huggingface.co/AXERA-TECH/Qwen3-VL-4B-Instruct-GPTQ-Int4)
- **Architecture**: Qwen3 VL 4B, W4A16 GPTQ quantization
- **Components**: 36 decoder layers (layer0 79MB), post (405MB), vision_u8 (422MB)
- **NPU**: 3-core mode
- **Benchmark**: axcl_run_model, warmup 3, repeat 10

### With vs Without Optimization

| Component | Optimized (ms) | Default (ms) | Improvement |
|-----------|:-:|:-:|:-:|
| Vision (u8) | 85.687 | 86.046 | **+0.4%** |
| LLM Layer (×36) | 3.431 | 3.688 | **+7.5%** |
| LLM Post | 18.842 | 19.181 | **+1.8%** |

### Estimated Decode Speed

- **Optimized**: 36 × 3.431 + 18.842 = 142.4ms → **7.0 tok/s**
- **Default**: 36 × 3.688 + 19.181 = 151.9ms → **6.6 tok/s**
- **Optimization gain**: +6.7%

### Analysis

Qwen3-VL-4B Int4 is the largest VLM tested with Int4 quantization. The 7.0 tok/s decode speed is reasonable for a 4B model — close to Qwen3-4B W8A16 native (7.42 tok/s). Vision encoder at 85.7ms is very fast for 4B VL, essentially no optimization effect (NPU-bound at this size). The 36 decoder layers at 3.4ms each show moderate +7.5% speedup — consistent with mid-size models in the pattern table.

## Methodology

### LLM

- Models tested: Qwen3-0.6B (W8A16), Qwen3-1.7B (W8A16), Qwen3-4B (W8A16), Qwen2.5-7B (W4A16 GPTQ)
- Each benchmark run: single prompt ("What is artificial intelligence? Answer briefly." or similar)
- TTFT measured by ax-llm runtime (Qwen3 binary logs TTFT, Qwen2.5 binary does not)
- Decode speed measured as average over full response generation
- Qwen3-0.6B: 3 runs per configuration, averaged (detailed results above)
- Larger models: measured until stable tok/s (typically 100+ tokens)
- "Cold" = first run after reboot (model loading from disk)
- "Warm" = subsequent runs (model in page cache)

### Vision, Classification, Face, Super-Resolution, Segmentation, Zero-Shot, Detection, Speech

- `axcl_run_model -m model.axmodel -r 100 -w 10` (100 repeats, 10 warmup)
- For heavy models (Real-ESRGAN 256, MobileSAM encoder): `-r 20-50 -w 3-5`
- Pure NPU inference time (no image loading, no post-processing)
- Pre-compiled models from [HuggingFace AXERA-TECH](https://huggingface.co/AXERA-TECH)
- Classification models compiled locally with Pulsar2 5.1, INT8 MinMax
- Default = IRQ on CPU0 (A55), schedutil governor
- Optimized = IRQ on CPU4 (A76), performance governor

### OCR

- NPU inference time measured via PyAXEngine (axengine) internal timer
- 100 runs averaged
- Full pipeline tested on real images with mixed Chinese/English text

### TTS (CosyVoice3)

- Custom C++ binary via AXCL with tokenizer server
- 3 runs per configuration, key metrics averaged
- Stochastic model (different token counts per run), so TTFT and decode speed compared, not RTF

### Video Transcode

- `axcl_sample_transcode` with 1080p H.264 input
- FPS measured over full video duration
- Tested with and without optimization, multiple `hwclk` settings
