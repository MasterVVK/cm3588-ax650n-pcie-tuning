# Benchmark Results

## Test Configuration

- **Model**: Qwen3-0.6B (W8A16, 28 layers)
- **Runtime**: AXCL aarch64 (ax-llm)
- **Device**: AX650N (M5Stack Module LLM / AI-8850)
- **Host**: CM3588 NAS (RK3588, 32GB RAM)
- **PCIe**: Gen2 x1 (500 MB/s)
- **Driver**: AXCL V3.6.4
- **Kernel**: 6.1.118
- **Date**: 2026-02-09

## Cross-Platform Comparison

| Platform | PCIe | Qwen3-0.6B tok/s | TTFT | Notes |
|----------|------|-------------------|------|-------|
| AX650N native | — | 19-20 | — | No PCIe overhead |
| RPi5 + M.2 HAT | Gen2 x1 | ~13 | — | BCM2712 |
| **CM3588 (optimized)** | **Gen2 x1** | **11-12.6** | **353-397 ms** | **This project** |
| CM3588 (default) | Gen2 x1 | 5-7.5 | 440-616 ms | No optimization |

## CM3588 Detailed Results

### Without Optimization (default after reboot)

| Run | TTFT (ms) | Decode (tok/s) | Notes |
|-----|-----------|----------------|-------|
| 1 (cold) | 590 | 5.02 | First run after reboot |
| 2 (warm) | 439 | 7.52 | |
| 3 (warm) | 616 | 5.33 | Frequency dropped |

**Average: ~5.96 tok/s** (high variance due to dynamic frequency scaling)

### With Optimization (IRQ affinity + performance governor)

| Run | TTFT (ms) | Decode (tok/s) | Notes |
|-----|-----------|----------------|-------|
| 1 | 358 | 12.34 | |
| 2 | 397 | 11.11 | |
| 3 | 353 | 12.62 | |

**Average: 12.02 tok/s** (stable, low variance)

### Improvement

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Decode speed (avg) | 5.96 tok/s | 12.02 tok/s | **+102%** |
| TTFT (best) | 439 ms | 353 ms | **-20%** |
| Stability | High variance | Stable | Consistent results |

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

The optimization effect on vision models (2-13%) is smaller than on LLM (+100%) because:

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

## Optimization Effect Pattern

The speedup from PCIe optimization correlates inversely with inference time:

| Inference time | Model | Speedup |
|:-:|:-:|:-:|
| < 0.5 ms | OCR classifier (cls) | **+71%** |
| ~0.7 ms | MobileNetV2 | **+50%** |
| ~1.4 ms | ResNet18 | **+37%** |
| ~3.5 ms | ResNet50 | +8% |
| ~7 ms | YOLOv5s | +5% |
| ~29 ms | OCR detector (det) | +1% |

**Why?** Each NPU inference involves PCIe round-trip overhead (~0.3ms for IRQ handling + data transfer). For fast models, this overhead is a significant fraction of total time. Moving IRQ to a faster CPU core (A76 @ 2.3 GHz vs A55 @ 1.8 GHz) reduces this overhead, and the `performance` governor eliminates frequency scaling delays between calls.

For LLM inference, the effect is even more dramatic (+100%) because each token requires hundreds of sequential small NPU calls, each incurring PCIe overhead.

## Methodology

### LLM

- Each benchmark run: single prompt ("What is 2+2?" or similar short prompt)
- TTFT measured by ax-llm runtime
- Decode speed measured as average over full response generation
- "Cold" = first run after reboot (model loading from disk)
- "Warm" = subsequent runs (model in page cache)

### Vision & Classification

- `axcl_run_model -m model.axmodel -r 100 -w 10` (100 repeats, 10 warmup)
- Pure NPU inference time (no image loading, no post-processing)
- Vision models from HuggingFace (Pulsar2 compiled, w8a16 quantization)
- Classification models compiled locally with Pulsar2 5.1, INT8 MinMax
- Default = IRQ on CPU0 (A55), schedutil governor
- Optimized = IRQ on CPU4 (A76), performance governor

### OCR

- NPU inference time measured via PyAXEngine (axengine) internal timer
- 100 runs averaged
- Full pipeline tested on real images with mixed Chinese/English text

### Video Transcode

- `axcl_sample_transcode` with 1080p H.264 input
- FPS measured over full video duration
- Tested with and without optimization, multiple `hwclk` settings
