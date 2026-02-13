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

## Optimization Effect Pattern

The speedup from PCIe optimization correlates inversely with inference time:

| Inference time | Model | Speedup |
|:-:|:-:|:-:|
| ~0.20 ms | LivePortrait stitching | **+57%** |
| ~0.24 ms | Zipformer decoder | **+80%** |
| ~0.27 ms | EdgeTAM prompt encoder | **+10%** |
| ~0.3 ms | Insightface genderage | **+34%** |
| ~0.34 ms | Zipformer joiner | **+93%** |
| < 0.5 ms | OCR classifier (cls) | **+71%** |
| ~0.57 ms | SmolVLM2-256M LLM layer | **+45%** |
| ~0.7 ms | MobileNetV2 | **+50%** |
| ~0.7 ms | EdgeTAM prompt mask | +4% |
| ~1.2 ms | FastVLM-0.5B LLM layer | **+32%** |
| ~1.4 ms | ResNet18 | **+37%** |
| ~1.4 ms | gtcrn (audio denoise) | +12% |
| ~1.6 ms | SmolVLM2-256M LLM post | +21% |
| ~1.6 ms | SATRN decoder | **+37%** |
| ~1.7 ms | YOLO26n-Pose | **+22%** |
| ~1.8 ms | Qwen3-Embedding layer | +12.5% |
| ~2.3 ms | YOLO26n-Seg | **+22%** |
| ~3.0 ms | Zipformer encoder | +19% |
| ~3.0 ms | YOLO-World CLIP | +9% |
| ~3.5 ms | ResNet50 | +8% |
| ~3.6 ms | QR YOLO26n/YOLO11n | +12% |
| ~3.7 ms | YOLO26s-Pose/Insightface w600k_r50 | +10-15% |
| ~3.9 ms | 3D-Speaker ECAPA-TDNN | +3% |
| ~4.0 ms | QR DEIMv2-femto | +9% |
| ~4.6 ms | LibCLIP cnclip text | +10% |
| ~5.0 ms | YOLO26s-Seg | +12% |
| ~5.2 ms | EdgeTAM mask decoder | +3% |
| ~5.5 ms | 3D-Speaker Res2NetV2 | +1% |
| ~7.0 ms | FastVLM-0.5B LLM post | +7% |
| ~7 ms | YOLOv5s/Insightface det | +5-7% |
| ~7.5 ms | LivePortrait motion | +9% |
| ~8.5 ms | MobileCLIP2-S0 image | +2% |
| ~9 ms | RT-DETR/YOLO-World YOLO | +2-5% |
| ~9.6 ms | YOLO26m-Pose | +7% |
| ~10.4 ms | MixFormerV2 (tracking) | +3% |
| ~11 ms | FG-CLIP text/SigLIP2 vision | +1-5% |
| ~12.2 ms | YOLO26l-Pose | +6% |
| ~12.4 ms | SenseVoice streaming | +6% |
| ~13 ms | YOLOv7-Face/DeepLabv3Plus/jina-clip text | +1-4% |
| ~16 ms | RealESRGAN-x2 (CodeFormer) | +2% |
| ~20 ms | LivePortrait feature | +3% |
| ~21 ms | Whisper encoder/RAFT-stereo | ~0-1% |
| ~23 ms | Depth-Anything-3 small | +3% |
| ~24 ms | EdgeTAM image encoder | +1% |
| ~26 ms | YOLO26x-Pose | +3% |
| ~28 ms | SuperPoint | +1% |
| ~29 ms | OCR detector (det)/YOLOv5l-Face | +1-2% |
| ~37 ms | YOLO26x-Seg | +2% |
| ~43 ms | DEIMv2 DINOv3-S | +1% |
| ~45 ms | FastVLM-0.5B vision encoder | +2% |
| ~51 ms | MobileSAM encoder | +1% |
| ~55 ms | SenseVoice (full) | +1% |
| ~65 ms | MobileCLIP2-S4 image | +1% |
| ~68 ms | Depth-Anything-3 base | +1% |
| ~89 ms | LibCLIP cnclip vision | +0.8% |
| ~99 ms | SmolVLM2-256M vision encoder | +1% |
| ~107 ms | RMBG-1.4 (background removal) | +1% |
| ~113 ms | RAFT-stereo 384x1280 | ~0% |
| ~129 ms | FG-CLIP image encoder | +0.4% |
| ~143 ms | IGEV++ (RTIGEV) | ~0% |
| ~210 ms | RIFE x2 720p (frame interp) | +0.4% |
| ~233 ms | LivePortrait spade | +0.3% |
| ~383 ms | DeOldify (colorization) | +0.2% |
| ~426 ms | mel_band_roformer (music sep) | +0.2% |
| ~445 ms | CodeFormer (face restoration) | +0.1% |
| ~475 ms | Real-ESRGAN 256→1024 | +0.2% |
| ~498 ms | DeOldify artistic | +0.2% |
| ~597 ms | jina-clip-v2 image encoder | +0.2% |

**Why?** Each NPU inference involves PCIe round-trip overhead (~0.3ms for IRQ handling + data transfer). For fast models, this overhead is a significant fraction of total time. Moving IRQ to a faster CPU core (A76 @ 2.3 GHz vs A55 @ 1.8 GHz) reduces this overhead, and the `performance` governor eliminates frequency scaling delays between calls.

For LLM inference, the effect is even more dramatic (+50-100%) because each token requires hundreds of sequential small NPU calls, each incurring PCIe overhead. Smaller, more efficient LLM architectures (MiniCPM4, SmolLM2) show the highest gains. VLM decoder layers show +32-45% improvement — consistent with the LLM pattern.

Zipformer joiner at **+93%** is the absolute record — beating OCR classifier (+71%) as the previous champion. The sub-0.5ms models consistently show the most dramatic speedups, confirming that PCIe round-trip latency is the dominant factor for ultra-fast inference.

**110+ models tested** across 28 categories confirm this pattern holds universally. For LLM, 9 configurations across 7 model families from 0.36B to 7B were tested, all showing significant speedup (+19% to +100%). VLM component benchmarks add 2 more model families. Portrait animation (LivePortrait) and streaming ASR (Zipformer) provide additional extreme data points.

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

Zipformer joiner at **+93%** is the **absolute record speedup** measured across all 110+ models — beating the previous record of +71% (OCR classifier). The decoder at +80% is the second highest.

Both joiner and decoder are ultra-fast sub-millisecond models where PCIe round-trip latency (~0.3ms) is a massive fraction of total time. Moving IRQ from slow A55 to fast A76 nearly halves the overhead.

The encoder at 3ms shows +19% — still a strong benefit, consistent with other models in this latency range.

For streaming ASR, the joiner and decoder are called once per frame (typically every 60ms), so the absolute time savings (~0.3ms + ~0.2ms per frame) are meaningful for low-latency applications.

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
