# Benchmark Results

## Test Configuration

- **Runtime**: AXCL aarch64 (ax-llm)
- **Device**: AX650N (M5Stack Module LLM / AI-8850)
- **Host**: CM3588 NAS (RK3588, 32GB RAM)
- **PCIe**: Gen2 x1 (500 MB/s)
- **Driver**: AXCL V3.6.4
- **Kernel**: 6.1.118
- **Date**: 2026-02-09 — 2026-02-11

## LLM Inference — Multi-Model Comparison

### Decode Speed (tok/s)

| Model | Quant | Layers | Size | Default | Optimized | Speedup | Native (official) |
|-------|-------|-------:|-----:|--------:|----------:|--------:|------------------:|
| Qwen3-0.6B | W8A16 | 28 | 1.0 GB | 7.1-7.5 | **10-12** | +50% | 19-20 |
| Qwen3-1.7B | W8A16 | 24 | 2.7 GB | 5.1-5.3 | **7.8-8.0** | +50% | 7.42 |
| Qwen3-4B | W8A16 | 36 | 5.1 GB | 2.6-2.8 | **3.7** | +37% | — |
| Qwen2.5-7B | W4A16 | 28 | 5.2 GB | 3.7 | **4.4** | +19% | 4.8 |

### TTFT (Time To First Token)

| Model | Default | Optimized | Speedup |
|-------|--------:|----------:|--------:|
| Qwen3-0.6B | 488-578 ms | **391 ms** | +25% |
| Qwen3-1.7B | 541 ms | **447 ms** | +21% |
| Qwen3-4B | 1216 ms | **1110 ms** | +10% |

*Note: Qwen2.5-7B binary does not log TTFT.*

### Key Observations

- **Qwen3-1.7B** optimized reaches **~108% of official native** (7.9 vs 7.42 tok/s) — likely measurement variance, but PCIe overhead is effectively eliminated for compute-bound models
- **Qwen2.5-7B** optimized reaches **92% of native** (4.4 vs 4.8 tok/s)
- Optimization effect is strongest for small models (+50% for 0.6B and 1.7B) where PCIe latency dominates
- Larger models are more compute-bound, so PCIe overhead is proportionally smaller, but optimization still provides +19-37%

### Cross-Platform Comparison (Qwen3-0.6B)

| Platform | PCIe | tok/s | TTFT | Notes |
|----------|------|------:|-----:|-------|
| AX650N native | — | 19-20 | — | No PCIe overhead |
| RPi5 + M.2 HAT | Gen2 x1 | ~13 | — | BCM2712 |
| **CM3588 (optimized)** | **Gen2 x1** | **10-12** | **391 ms** | **This project** |
| CM3588 (default) | Gen2 x1 | 7.1-7.5 | 488-578 ms | No optimization |

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
| ~0.27 ms | EdgeTAM prompt encoder | **+10%** |
| ~0.3 ms | Insightface genderage | **+34%** |
| < 0.5 ms | OCR classifier (cls) | **+71%** |
| ~0.7 ms | MobileNetV2 | **+50%** |
| ~0.7 ms | EdgeTAM prompt mask | +4% |
| ~1.4 ms | ResNet18 | **+37%** |
| ~1.4 ms | gtcrn (audio denoise) | +12% |
| ~3.0 ms | YOLO-World CLIP | +9% |
| ~3.5 ms | ResNet50 | +8% |
| ~3.7 ms | Insightface w600k_r50 | +15% |
| ~3.9 ms | 3D-Speaker ECAPA-TDNN | +3% |
| ~5.2 ms | EdgeTAM mask decoder | +3% |
| ~5.5 ms | 3D-Speaker Res2NetV2 | +1% |
| ~7 ms | YOLOv5s/Insightface det | +5-7% |
| ~9 ms | RT-DETR/YOLO-World YOLO | +2-5% |
| ~11 ms | SigLIP2 vision | +1% |
| ~21 ms | Whisper encoder/RAFT-stereo | ~0-1% |
| ~24 ms | EdgeTAM image encoder | +1% |
| ~29 ms | OCR detector (det) | +1% |
| ~51 ms | MobileSAM encoder | +1% |
| ~113 ms | RAFT-stereo 384x1280 | ~0% |
| ~143 ms | IGEV++ (RTIGEV) | ~0% |
| ~475 ms | Real-ESRGAN 256→1024 | +0.2% |

**Why?** Each NPU inference involves PCIe round-trip overhead (~0.3ms for IRQ handling + data transfer). For fast models, this overhead is a significant fraction of total time. Moving IRQ to a faster CPU core (A76 @ 2.3 GHz vs A55 @ 1.8 GHz) reduces this overhead, and the `performance` governor eliminates frequency scaling delays between calls.

For LLM inference, the effect is even more dramatic (+50%) because each token requires hundreds of sequential small NPU calls, each incurring PCIe overhead.

**40+ models tested** across 13 categories confirm this pattern holds universally. For LLM, 4 model sizes from 0.6B to 7B were tested, all showing significant speedup (+19% to +50%).

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
