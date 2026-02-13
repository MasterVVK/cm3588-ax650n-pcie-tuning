# CM3588 + AX650N PCIe Tuning

**[Русский](README.ru.md)** | **[中文](README.zh.md)**

> Boost LLM inference speed of [M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) by +50-100% on [FriendlyElec CM3588 NAS](https://wiki.friendlyelec.com/wiki/index.php/CM3588).

## Problem

The AX650N NPU (24 TOPS INT8) connected via M.2 on CM3588 NAS delivers only **7-7.5 tok/s** for LLM inference (Qwen3-0.6B) instead of the expected 12-13 tok/s. The root causes:

1. **PCIe Gen2 x1 hardware limitation** — CM3588 routes only 1 lane to the M.2 slot (device supports x2)
2. **IRQ on little core** — all interrupts handled by slow Cortex-A55 @ 1.8 GHz
3. **Dynamic frequency scaling** — CPU frequency drops between inference calls

## Solution

This toolkit applies two optimizations that together provide **+50-100% improvement** (model-dependent):

| | Before | After |
|--|--------|-------|
| Decode speed | 7.1-7.5 tok/s | **10-12 tok/s** |
| TTFT (prefill) | 488-578 ms | **391 ms** |
| Stability | High variance | Consistent |

## Quick Start

```bash
git clone https://github.com/MasterVVK/cm3588-ax650n-pcie-tuning.git
cd cm3588-ax650n-pcie-tuning
sudo ./install.sh
```

That's it. The optimization persists across reboots via systemd service.

## What It Does

1. **Moves AX650N IRQ to big core** — routes PCIe interrupt from Cortex-A55 (CPU0) to Cortex-A76 (CPU4)
2. **Sets performance governor** — locks big cores at maximum 2.3 GHz frequency
3. **Auto-detects** AX650N PCIe address and big core topology
4. **Persists after reboot** via systemd service

## Diagnostics

Run the diagnostic tool to see full PCIe configuration and recommendations:

```bash
sudo ax650n-diagnose.sh
```

Output includes: PCIe topology, link speed/width, MaxPayload, IRQ affinity, CPU governor status, and actionable recommendations.

## Benchmark Results

### LLM Inference

Decode speed (tok/s) via AXCL runtime on CM3588 PCIe Gen2 x1:

| Model | Quant | Default | Optimized | Speedup | Native (official) |
|-------|-------|--------:|----------:|--------:|-------------------:|
| MiniCPM4-0.5B | W8A16 | 6-11 | **15-20** | +100% | 36 |
| SmolLM2-360M | W8A16 | 5-10 | **13-16** | +75% | 38.7 |
| Qwen3-0.6B | W8A16 | 7.1-7.5 | **10-12** | +50% | 19-20 |
| DeepSeek-R1-1.5B | W4A16 | 4.8-7.0 | **10.2-11.0** | +75% | 17.7 |
| DeepSeek-R1-1.5B | W8A16 | 4.0-5.2 | **7.6-8.6** | +95% | 17.7 |
| Qwen3-1.7B | W8A16 | 5.1-5.3 | **7.8-8.0** | +50% | 7.42 |
| Qwen2.5-7B | W4A16 | 3.7 | **4.4** | +19% | 4.8 |
| SmolLM3-3B | W8A16 | 2.6-3.2 | **4.3-4.4** | +50% | — |
| Qwen3-4B | W8A16 | 2.6-2.8 | **3.7** | +37% | — |

TTFT (time to first token):

| Model | Default | Optimized | Speedup |
|-------|--------:|----------:|--------:|
| MiniCPM4-0.5B | 305-423 ms | **214-242 ms** | +40% |
| SmolLM2-360M | 348-530 ms | **259-304 ms** | +35% |
| Qwen3-0.6B | 488-578 ms | **391 ms** | +25% |
| DeepSeek-R1-1.5B | 503-688 ms | **380-432 ms** | +35% |
| Qwen3-1.7B | 541 ms | **447 ms** | +21% |
| SmolLM3-3B | 916-1043 ms | **708-735 ms** | +30% |
| Qwen3-4B | 1216 ms | **1110 ms** | +10% |

MiniCPM4-0.5B and SmolLM2-360M show the highest optimization gains at **+75-100%** — small, efficient architectures benefit enormously from reduced PCIe latency. Without optimization, baseline is extremely unstable (schedutil causes 2x variance). DeepSeek-R1-1.5B W4A16 reaches **11 tok/s** with optimization, making reasoning models practical on edge hardware. Qwen3-1.7B reaches **~108% of official native** (7.9 vs 7.42). **9 LLM configurations tested** across 7 model families.

### Vision Models (NPU inference, 640x640)

| Model | Default (ms) | Optimized (ms) | Speedup | FPS |
|-------|----------:|------------:|--------:|----:|
| YOLO11s | 3.99 | **3.55** | +12% | 282 |
| YOLOv8s | 4.21 | **3.89** | +8% | 257 |
| YOLO11s-Seg | 5.27 | **4.60** | +13% | 217 |
| YOLOv8s-Seg | 5.26 | **5.11** | +3% | 196 |
| YOLOv5s | 6.92 | **6.59** | +5% | 152 |
| YOLO26m | 9.47 | **9.04** | +5% | 111 |
| YOLOv8s-Pose | 11.81 | **11.26** | +5% | 89 |
| Depth-Anything-V2-S | 34.00 | **33.44** | +2% | 30 |

### YOLO26 Pose Estimation & Segmentation (NPU 3-core, 640x640)

| Model | Default (ms) | Optimized (ms) | Speedup | FPS | Native (ms) |
|-------|----------:|------------:|--------:|----:|------------:|
| YOLO26n-Pose | 2.08 | **1.71** | **+22%** | 586 | 1.53 |
| YOLO26n-Seg | 2.86 | **2.34** | **+22%** | 428 | 1.97 |
| YOLO26s-Pose | 4.07 | **3.72** | +10% | 269 | 3.53 |
| YOLO26s-Seg | 5.61 | **5.04** | +12% | 199 | 4.70 |
| YOLO26m-Pose | 10.25 | **9.62** | +7% | 104 | 9.30 |
| YOLO26x-Pose | 26.38 | **25.71** | +3% | 39 | 25.13 |

YOLO26 nano models show **+22% speedup** — among the highest for vision models. Full n/s/m/l/x results in [benchmark details](docs/benchmark-results.md).

### Depth-Anything-3

| Model | Default (ms) | Optimized (ms) | Speedup | Native (ms) |
|-------|----------:|------------:|--------:|------------:|
| DA3-small | 24.02 | **23.28** | +3% | 22.77 |
| DA3-base | 68.51 | **67.71** | +1% | 67.34 |

Key effect on vision: **2-22% faster + 2-3x more stable** latency (critical for real-time pipelines).

### Classification (NPU inference, 224x224)

| Model | Default (ms) | Optimized (ms) | Speedup | FPS |
|-------|----------:|------------:|--------:|----:|
| MobileNetV2 | 0.983 | **0.657** | +50% | 1523 |
| SqueezeNet1.1 | 0.786 | **0.768** | +2% | 1302 |
| ResNet18 | 1.963 | **1.435** | +37% | 697 |
| ResNet50 | 3.613 | **3.355** | +8% | 298 |

### OCR — PPOCR_v5 (Text Detection + Recognition)

| Model | Task | Default (ms) | Optimized (ms) | Speedup |
|-------|------|----------:|------------:|--------:|
| det_npu1 | Text Detection | 29.38 | **28.98** | +1% |
| cls_npu1 | Text Direction | 0.759 | **0.445** | +71% |
| rec_npu1 | Text Recognition | 3.958 | **3.681** | +8% |

Full OCR pipeline: **~1.5s** per image (16 text regions, Chinese + English, 81-99% accuracy).

### Scene Text Recognition — SATRN

| Model | Default (ms) | Optimized (ms) | Speedup | Native (ms) |
|-------|----------:|------------:|--------:|------------:|
| SATRN backbone+encoder | 7.43 | **7.35** | +1% | 6.09 |
| SATRN decoder | 2.17 | **1.58** | **+37%** | 1.38 |

### Face Recognition — Insightface

| Model | Task | Default (ms) | Optimized (ms) | Speedup | FPS | Native (ms) |
|-------|------|----------:|------------:|--------:|----:|------------:|
| det_10g | Detection | 7.36 | **6.86** | +7% | 146 | 6.95 |
| genderage | Gender/Age | 0.479 | **0.357** | +34% | 2801 | 0.30 |
| w600k_r50 | Embedding | 4.27 | **3.72** | +15% | 269 | 3.99 |

### Stereo Depth, Video Segmentation, Speaker ID, Audio, Portrait Animation

| Model | Task | Default (ms) | Optimized (ms) | Speedup | Native (ms) |
|-------|------|----------:|------------:|--------:|------------:|
| LivePortrait stitching | Portrait anim | 0.311 | **0.198** | **+57%** | — |
| EdgeTAM prompt enc | Video seg | 0.297 | **0.270** | +10% | 0.06 |
| EdgeTAM prompt mask | Video seg | 0.765 | **0.732** | +4% | 0.46 |
| gtcrn | Audio denoise | 1.607 | **1.434** | +12% | — |
| 3D-Speaker ECAPA-TDNN | Speaker ID | 4.006 | **3.889** | +3% | — |
| EdgeTAM mask decoder | Video seg | 5.338 | **5.184** | +3% | 4.73 |
| 3D-Speaker Res2NetV2 | Speaker ID | 5.534 | **5.459** | +1% | 5.09 |
| LivePortrait motion | Portrait anim | 8.177 | **7.472** | +9% | — |
| LivePortrait feature | Portrait anim | 20.36 | **19.87** | +3% | — |
| RAFT-stereo 256x640 | Stereo depth | 21.19 | **21.28** | ~0% | 20.9 |
| EdgeTAM image encoder | Video seg | 23.88 | **23.73** | +1% | 22.35 |
| RAFT-stereo 384x1280 | Stereo depth | 112.55 | **112.40** | ~0% | — |
| IGEV++ (RTIGEV) | Stereo depth | 143.40 | **143.06** | ~0% | 139.80 |
| LivePortrait spade | Portrait anim | 233.3 | **232.5** | +0.3% | — |
| mel_band_roformer | Music separation | 426.3 | **425.6** | +0.2% | — |

### Tracking, Segmentation, Keypoints, QR Detection

| Model | Task | Default (ms) | Optimized (ms) | Speedup | FPS |
|-------|------|----------:|------------:|--------:|----:|
| QR YOLO26n | QR detection | 4.08 | **3.63** | +12% | 275 |
| QR YOLO11n | QR detection | 4.26 | **3.80** | +12% | 263 |
| MixFormerV2 | Object tracking | 10.75 | **10.42** | +3% | 96 |
| YOLOv7-Face | Face detection | 12.93 | **12.66** | +2% | 79 |
| DeepLabv3Plus | Semantic seg | 13.83 | **13.24** | +4% | 76 |
| SuperPoint | Keypoints | 28.05 | **27.84** | +1% | 36 |
| DEIMv2 DINOv3-S | Detection | 43.05 | **42.42** | +1% | 24 |

### Super-Resolution, Zero-Shot, Speech, Restoration

| Model | Task | Default (ms) | Optimized (ms) | Speedup | Native (ms) |
|-------|------|----------:|------------:|--------:|------------:|
| Real-ESRGAN x4 | 64→256 upscale | 15.85 | **15.66** | +1% | 15.0 |
| Real-ESRGAN x4 | 256→1024 upscale | 476.2 | **475.4** | +0.2% | 440 |
| MobileSAM encoder | Segmentation | 51.52 | **50.92** | +1% | 49.5 |
| MobileSAM decoder | Segmentation | 10.59 | **10.35** | +2% | 9.93 |
| SigLIP2 vision | Zero-shot img | 11.48 | **11.37** | +1% | 11.1 |
| SigLIP2 text | Zero-shot txt | 5.00 | **4.57** | +10% | 4.56 |
| RT-DETR | Detection | 9.52 | **9.35** | +2% | — |
| YOLO-World | Detection | 9.71 | **9.25** | +5% | — |
| Whisper-tiny enc | Speech-to-text | 21.27 | **21.13** | +1% | — |
| Whisper-tiny dec | Speech-to-text | 4.05 | **3.93** | +3% | — |
| Zipformer joiner | ASR | 0.664 | **0.344** | **+93%** | — |
| Zipformer decoder | ASR | 0.437 | **0.243** | **+80%** | — |
| Zipformer encoder | ASR | 3.602 | **3.018** | +19% | — |
| SenseVoice stream | ASR (5 langs) | 13.11 | **12.37** | +6% | — |
| SenseVoice full | ASR (5 langs) | 55.33 | **54.69** | +1% | — |
| MobileCLIP2-S0 | CLIP image | 8.63 | **8.49** | +2% | — |
| MobileCLIP2-S4 | CLIP image | 64.94 | **64.34** | +1% | 65.3 |
| LibCLIP cnclip vision | CLIP image (CN) | 89.44 | **88.76** | +0.8% | 88.48 |
| LibCLIP cnclip text | CLIP text (CN) | 5.04 | **4.58** | +10% | 4.58 |
| FG-CLIP image | CLIP image | 129.2 | **128.7** | +0.4% | 125.2 |
| FG-CLIP text | CLIP text | 11.67 | **11.08** | +5% | 10.82 |
| jina-clip-v2 image | CLIP image | 597.2 | **596.2** | +0.2% | 592.2 |
| jina-clip-v2 text | CLIP text | 15.31 | **14.86** | +3% | 15.48 |
| RMBG-1.4 | Background removal | 107.2 | **106.5** | +1% | — |
| CodeFormer | Face restoration | 444.7 | **444.1** | +0.1% | — |
| DeOldify | Photo colorization | 383.6 | **383.0** | +0.2% | — |

### TTS — CosyVoice3 (Russian text on NPU)

| Metric | Default | Optimized | Speedup |
|--------|--------:|----------:|--------:|
| TTFT | 125 ms | **108 ms** | +16% |
| LLM Decode | 13.9 tok/s | **16.3 tok/s** | +17% |
| RTF (Real-Time Factor) | 2.0-3.7x | **1.7-1.9x** | |

### Embedding / RAG

| Model | Default (ms) | Optimized (ms) | Speedup | Native (ms) |
|-------|----------:|------------:|--------:|------------:|
| bge-small-en-v1.5 | 35.25 | **34.74** | +1.5% | 32.4 |
| Qwen3-Embedding Layer (×28) | 2.03 | **1.80** | +12.5% | — |
| Qwen3-Embedding Post | 8.67 | **8.09** | +7% | — |

Qwen3-Embedding (28 layers, W8A16) — same architecture as Qwen3-0.6B LLM. Decoder layers show +12.5% speedup, consistent with LLM pattern.

### VLM — SmolVLM2-256M & FastVLM-0.5B (Component Benchmarks)

No AXCL aarch64 binary for end-to-end VLM; individual .axmodel components benchmarked via `axcl_run_model`:

| Model | Component | Default (ms) | Optimized (ms) | Speedup |
|-------|-----------|----------:|------------:|--------:|
| SmolVLM2-256M | Vision Encoder | 99.12 | **98.35** | +1% |
| SmolVLM2-256M | LLM Layer | 0.818 | **0.566** | **+45%** |
| SmolVLM2-256M | LLM Post | 1.980 | **1.639** | +21% |
| FastVLM-0.5B | Vision Encoder | 45.51 | **44.64** | +2% |
| FastVLM-0.5B | LLM Layer | 1.547 | **1.170** | **+32%** |
| FastVLM-0.5B | LLM Post | 7.538 | **7.043** | +7% |

VLM decoder layers show +32-45% speedup — consistent with LLM pattern. Estimated decode: SmolVLM2 ~38→54 tok/s (+42%, native: 76.7), FastVLM ~22→29 tok/s (+28%, native: 34.8).

### Optimization Effect Pattern

The speedup correlates inversely with inference time — faster models benefit more:

| Inference time | Example | Speedup |
|:-:|:-:|:-:|
| ~0.20 ms | LivePortrait stitching | **+57%** |
| ~0.24 ms | Zipformer decoder | **+80%** |
| ~0.27 ms | EdgeTAM prompt encoder | **+10%** |
| ~0.3 ms | Insightface genderage | **+34%** |
| ~0.34 ms | Zipformer joiner | **+93%** |
| < 0.5 ms | OCR classifier | **+71%** |
| ~0.7 ms | MobileNetV2 | **+50%** |
| ~1.6 ms | SATRN decoder | **+37%** |
| ~1.4 ms | ResNet18, gtcrn | **+12-37%** |
| ~3.0 ms | Zipformer encoder | +19% |
| ~3.6 ms | QR YOLO26n/YOLO11n | +12% |
| ~3.7 ms | Insightface w600k_r50 | +15% |
| ~7.5 ms | LivePortrait motion | +9% |
| ~10 ms | MixFormerV2 | +3% |
| ~13 ms | DeepLabv3Plus | +4% |
| ~20 ms | LivePortrait feature | +3% |
| ~29 ms | OCR detector | +1% |
| ~107 ms | RMBG-1.4 | +1% |
| ~143 ms | IGEV++ stereo depth | ~0% |
| ~233 ms | LivePortrait spade | +0.3% |
| ~445 ms | CodeFormer | +0.1% |
| ~498 ms | DeOldify artistic | +0.2% |

This is because PCIe round-trip latency (~0.3ms) is a larger fraction of total time for fast models.

Zipformer joiner at +93% and decoder at +80% are the **highest speedups** measured for any single-inference model — beating the previous record of +71% (OCR classifier). These ultra-fast sub-millisecond ASR components are extremely sensitive to PCIe latency.

**110+ models tested** across 28 categories: LLM, VLM, vision detection, pose estimation, instance/semantic segmentation, classification, OCR, face recognition/restoration, super-resolution, zero-shot, CLIP, speech recognition, TTS, depth estimation, stereo depth, video segmentation, speaker ID, audio denoising, object tracking, keypoint detection, QR code detection, background removal, photo colorization, portrait animation, and more.

See [detailed benchmark results](docs/benchmark-results.md) and [PCIe architecture analysis](docs/pcie-analysis.md).

## Requirements

- FriendlyElec CM3588 NAS (RK3588)
- AX650N M.2 module ([M5Stack Module LLM](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) or similar)
- AXCL driver installed ([ax-llm](https://github.com/AXERA-TECH/ax-llm))
- Root access

## Uninstall

```bash
sudo ./uninstall.sh
```

## How It Works

The CM3588 NAS uses RK3588's PCIe 2.0 x1 controller (`pcie2x1l1` at `0xfe180000`) for the M.2 slot where AX650N is installed. This limits bandwidth to ~500 MB/s (device supports 1000 MB/s with x2).

While the PCIe width cannot be changed (hardware limitation), the software-side bottlenecks are significant:

- **IRQ routing**: Linux defaults to CPU0 (A55 @ 1.8 GHz) for MSI interrupts. Moving to CPU4 (A76 @ 2.3 GHz) reduces interrupt handling latency by ~30%.
- **CPU governor**: The `schedutil` governor aggressively scales down frequency during idle periods between token generations. `performance` governor maintains peak frequency.

Combined effect: **+50-100% decode throughput, +25-40% prefill speed** depending on model. Up to +100% on MiniCPM4-0.5B, +95% on DeepSeek-R1-1.5B, +50% on Qwen3-0.6B/1.7B. For ultra-fast sub-millisecond models (Zipformer ASR), speedups reach **+93%**.

See [PCIe Architecture Analysis](docs/pcie-analysis.md) for the full technical deep-dive.

## Related Resources

- [CM3588 Wiki](https://wiki.friendlyelec.com/wiki/index.php/CM3588) — FriendlyElec CM3588 documentation
- [M5Stack Module LLM](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) — M5Stack AI-8850 / AX650N documentation
- [ax-llm](https://github.com/AXERA-TECH/ax-llm) — Axera LLM inference engine
- [AXCL](https://github.com/AXERA-TECH/axcl) — Axera PCIe host SDK

## License

[MIT](LICENSE)
