# CM3588 + AX650N PCIe Tuning

**[Русский](README.ru.md)** | **[中文](README.zh.md)**

> Double the LLM inference speed of [M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) on [FriendlyElec CM3588 NAS](https://wiki.friendlyelec.com/wiki/index.php/CM3588).

## Problem

The AX650N NPU (24 TOPS INT8) connected via M.2 on CM3588 NAS delivers only **5-7 tok/s** for LLM inference instead of the expected 12-13 tok/s. The root causes:

1. **PCIe Gen2 x1 hardware limitation** — CM3588 routes only 1 lane to the M.2 slot (device supports x2)
2. **IRQ on little core** — all interrupts handled by slow Cortex-A55 @ 1.8 GHz
3. **Dynamic frequency scaling** — CPU frequency drops between inference calls

## Solution

This toolkit applies two optimizations that together provide **+100% improvement**:

| | Before | After |
|--|--------|-------|
| Decode speed | 5-7.5 tok/s | **11-12.6 tok/s** |
| TTFT (prefill) | 440-616 ms | **353-397 ms** |
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

Model: Qwen3-0.6B (W8A16) via AXCL runtime

| Platform | PCIe | tok/s |
|----------|------|-------|
| AX650N native | — | 19-20 |
| Raspberry Pi 5 | Gen2 x1 | ~13 |
| **CM3588 (optimized)** | **Gen2 x1** | **12.0** |
| CM3588 (default) | Gen2 x1 | 5-7 |

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

Key effect on vision: **2-13% faster + 2-3x more stable** latency (critical for real-time pipelines).

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

### Face Recognition — Insightface

| Model | Task | Default (ms) | Optimized (ms) | Speedup | FPS |
|-------|------|----------:|------------:|--------:|----:|
| det_10g | Detection | 7.36 | **6.86** | +7% | 146 |
| genderage | Gender/Age | 0.479 | **0.357** | +34% | 2801 |
| w600k_r50 | Embedding | 4.27 | **3.72** | +15% | 269 |

### Super-Resolution, Segmentation, Zero-Shot, Speech

| Model | Task | Default (ms) | Optimized (ms) | Speedup |
|-------|------|----------:|------------:|--------:|
| Real-ESRGAN x4 | 64→256 upscale | 15.85 | **15.66** | +1% |
| Real-ESRGAN x4 | 256→1024 upscale | 476.2 | **475.4** | +0.2% |
| MobileSAM encoder | Segmentation | 51.52 | **50.92** | +1% |
| MobileSAM decoder | Segmentation | 10.59 | **10.35** | +2% |
| SigLIP2 vision | Zero-shot img | 11.48 | **11.37** | +1% |
| SigLIP2 text | Zero-shot txt | 5.00 | **4.57** | +10% |
| RT-DETR | Detection | 9.52 | **9.35** | +2% |
| YOLO-World | Detection | 9.71 | **9.25** | +5% |
| Whisper-tiny enc | Speech-to-text | 21.27 | **21.13** | +1% |
| Whisper-tiny dec | Speech-to-text | 4.05 | **3.93** | +3% |

### TTS — CosyVoice3 (Russian text on NPU)

| Metric | Default | Optimized | Speedup |
|--------|--------:|----------:|--------:|
| TTFT | 125 ms | **108 ms** | +16% |
| LLM Decode | 13.9 tok/s | **16.3 tok/s** | +17% |
| RTF (Real-Time Factor) | 2.0-3.7x | **1.7-1.9x** | |

### Optimization Effect Pattern

The speedup correlates inversely with inference time — faster models benefit more:

| Inference time | Example | Speedup |
|:-:|:-:|:-:|
| ~0.3 ms | Insightface genderage | **+34%** |
| < 0.5 ms | OCR classifier | **+71%** |
| ~0.7 ms | MobileNetV2 | **+50%** |
| ~1.4 ms | ResNet18 | **+37%** |
| ~3.7 ms | Insightface w600k_r50 | +15% |
| ~3.5 ms | ResNet50 | +8% |
| ~7 ms | YOLOv5s | +5% |
| ~29 ms | OCR detector | +1% |
| ~475 ms | Real-ESRGAN 256→1024 | +0.2% |

This is because PCIe round-trip latency (~0.3ms) is a larger fraction of total time for fast models.

**30+ models tested** across 10 categories: LLM, vision detection, segmentation, classification, OCR, face recognition, super-resolution, zero-shot, speech recognition, and TTS.

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

Combined effect: **+100% decode throughput, +20% prefill speed**.

See [PCIe Architecture Analysis](docs/pcie-analysis.md) for the full technical deep-dive.

## Related Resources

- [CM3588 Wiki](https://wiki.friendlyelec.com/wiki/index.php/CM3588) — FriendlyElec CM3588 documentation
- [M5Stack Module LLM](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) — M5Stack AI-8850 / AX650N documentation
- [ax-llm](https://github.com/AXERA-TECH/ax-llm) — Axera LLM inference engine
- [AXCL](https://github.com/AXERA-TECH/axcl) — Axera PCIe host SDK

## License

[MIT](LICENSE)
