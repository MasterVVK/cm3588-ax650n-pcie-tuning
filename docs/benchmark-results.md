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

## Vision Methodology

- `axcl_run_model -m model.axmodel -r 100 -w 10` (100 repeats, 10 warmup)
- Pure NPU inference time (no image loading, no post-processing)
- Models from HuggingFace (Pulsar2 compiled, w8a16 quantization)
- Default = IRQ on CPU0 (A55), schedutil governor
- Optimized = IRQ on CPU4 (A76), performance governor
