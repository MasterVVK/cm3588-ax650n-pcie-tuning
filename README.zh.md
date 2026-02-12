# CM3588 + AX650N PCIe 性能优化

**[English](README.md)** | **[Русский](README.ru.md)**

> 在 [FriendlyElec CM3588 NAS](https://wiki.friendlyelec.com/wiki/index.php/CM3588) 上将 [M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) 的 LLM 推理速度提升 50%。

## 问题

AX650N NPU（24 TOPS INT8）通过 M.2 连接到 CM3588 NAS，LLM 推理速度仅有 **7-7.5 tok/s**（Qwen3-0.6B），低于预期的 12-13 tok/s。根本原因：

1. **PCIe Gen2 x1 硬件限制** — CM3588 仅为 M.2 插槽布线了 1 条通道（设备支持 x2）
2. **中断在小核上处理** — 所有中断由低速 Cortex-A55 @ 1.8 GHz 处理
3. **动态频率调节** — CPU 在推理调用间隙降低频率

## 解决方案

本工具包应用两项优化，总共带来 **+50% 性能提升**（Qwen3-0.6B）：

| | 优化前 | 优化后 |
|--|--------|--------|
| 解码速度 | 7.1-7.5 tok/s | **10-12 tok/s** |
| 首字延迟 (TTFT) | 488-578 ms | **391 ms** |
| 稳定性 | 波动大 | 稳定 |

## 快速开始

```bash
git clone https://github.com/MasterVVK/cm3588-ax650n-pcie-tuning.git
cd cm3588-ax650n-pcie-tuning
sudo ./install.sh
```

通过 systemd 服务实现重启后自动应用优化。

## 功能说明

1. **将 AX650N 中断转移到大核** — 从 Cortex-A55 (CPU0) 转移到 Cortex-A76 (CPU4)
2. **设置 performance 调频策略** — 将大核锁定在最高 2.3 GHz
3. **自动检测** AX650N PCIe 地址和 CPU 拓扑
4. **重启后自动生效** — 通过 systemd 服务实现

## 诊断工具

```bash
sudo ax650n-diagnose.sh
```

输出包括：PCIe 拓扑、链路速度/宽度、MaxPayload、中断亲和性、CPU 调频策略及优化建议。

## 基准测试结果

### LLM 推理

解码速度 (tok/s)，通过 AXCL 运行时，CM3588 PCIe Gen2 x1：

| 模型 | 量化 | 默认 | 优化后 | 提升 | Native（官方） |
|------|------|-----:|-------:|-----:|--------------:|
| MiniCPM4-0.5B | W8A16 | 8.5-9.4 | **15.4-18.4** | +86% | 36 |
| SmolLM2-360M | W8A16 | 7.0-9.6 | **12.2-14.0** | +63% | 38.7 |
| Qwen3-0.6B | W8A16 | 7.1-7.5 | **10-12** | +50% | 19-20 |
| DeepSeek-R1-1.5B | W4A16 | 4.9-6.5 | **10.2-11.0** | +70% | 17.7 |
| DeepSeek-R1-1.5B | W8A16 | 4.2-5.2 | **7.6-8.3** | +60% | 17.7 |
| Qwen3-1.7B | W8A16 | 5.1-5.3 | **7.8-8.0** | +50% | 7.42 |
| Qwen2.5-7B | W4A16 | 3.7 | **4.4** | +19% | 4.8 |
| SmolLM3-3B | W8A16 | 2.6-3.2 | **4.3-4.4** | +50% | — |
| Qwen3-4B | W8A16 | 2.6-2.8 | **3.7** | +37% | — |

首字延迟 (TTFT)：

| 模型 | 默认 | 优化后 | 提升 |
|------|-----:|-------:|-----:|
| MiniCPM4-0.5B | 318-350 ms | **234-244 ms** | +30% |
| SmolLM2-360M | 347-373 ms | **285-304 ms** | +20% |
| Qwen3-0.6B | 488-578 ms | **391 ms** | +25% |
| DeepSeek-R1-1.5B | 509-661 ms | **380-432 ms** | +35% |
| Qwen3-1.7B | 541 ms | **447 ms** | +21% |
| SmolLM3-3B | 916-1043 ms | **708-735 ms** | +30% |
| Qwen3-4B | 1216 ms | **1110 ms** | +10% |

MiniCPM4-0.5B 展示了最高的优化增益 **+86%** — 其高效架构从 PCIe 延迟降低中获益最大。DeepSeek-R1-1.5B W4A16 优化后达到 **11 tok/s**，使推理模型在边缘硬件上变得实用。Qwen3-1.7B 优化后达到 **官方原生性能的约 108%**（7.9 vs 7.42）。**测试了 9 种 LLM 配置**，涵盖 7 个模型系列。

### 视觉模型（NPU 推理，640x640）

| 模型 | 默认 (ms) | 优化后 (ms) | 提升 | FPS |
|------|-------:|--------:|-----:|----:|
| YOLO11s | 3.99 | **3.55** | +12% | 282 |
| YOLOv8s | 4.21 | **3.89** | +8% | 257 |
| YOLO11s-Seg | 5.27 | **4.60** | +13% | 217 |
| YOLOv8s-Seg | 5.26 | **5.11** | +3% | 196 |
| YOLOv5s | 6.92 | **6.59** | +5% | 152 |
| YOLO26m | 9.47 | **9.04** | +5% | 111 |
| YOLOv8s-Pose | 11.81 | **11.26** | +5% | 89 |
| Depth-Anything-V2-S | 34.00 | **33.44** | +2% | 30 |

视觉模型关键效果：**速度提升 2-13% + 延迟稳定性提高 2-3 倍**（对实时管线至关重要）。

### 分类模型（NPU 推理，224x224）

| 模型 | 默认 (ms) | 优化后 (ms) | 提升 | FPS |
|------|-------:|--------:|-----:|----:|
| MobileNetV2 | 0.983 | **0.657** | +50% | 1523 |
| SqueezeNet1.1 | 0.786 | **0.768** | +2% | 1302 |
| ResNet18 | 1.963 | **1.435** | +37% | 697 |
| ResNet50 | 3.613 | **3.355** | +8% | 298 |

### OCR — PPOCR_v5（文字检测 + 识别）

| 模型 | 任务 | 默认 (ms) | 优化后 (ms) | 提升 |
|------|------|-------:|--------:|-----:|
| det_npu1 | 文字检测 | 29.38 | **28.98** | +1% |
| cls_npu1 | 文字方向 | 0.759 | **0.445** | +71% |
| rec_npu1 | 文字识别 | 3.958 | **3.681** | +8% |

### 人脸识别 — Insightface

| 模型 | 任务 | 默认 (ms) | 优化后 (ms) | 提升 | FPS |
|------|------|-------:|--------:|-----:|----:|
| det_10g | 人脸检测 | 7.36 | **6.86** | +7% | 146 |
| genderage | 性别/年龄 | 0.479 | **0.357** | +34% | 2801 |
| w600k_r50 | 特征提取 | 4.27 | **3.72** | +15% | 269 |

### 立体深度、视频分割、说话人识别、音频

| 模型 | 任务 | 默认 (ms) | 优化后 (ms) | 提升 |
|------|------|-------:|--------:|-----:|
| EdgeTAM prompt enc | 视频分割 | 0.297 | **0.270** | +10% |
| EdgeTAM prompt mask | 视频分割 | 0.765 | **0.732** | +4% |
| gtcrn | 音频降噪 | 1.607 | **1.434** | +12% |
| 3D-Speaker ECAPA-TDNN | 说话人识别 | 4.006 | **3.889** | +3% |
| EdgeTAM mask decoder | 视频分割 | 5.338 | **5.184** | +3% |
| 3D-Speaker Res2NetV2 | 说话人识别 | 5.534 | **5.459** | +1% |
| RAFT-stereo 256x640 | 立体深度 | 21.19 | **21.28** | ~0% |
| EdgeTAM image encoder | 视频分割 | 23.88 | **23.73** | +1% |
| RAFT-stereo 384x1280 | 立体深度 | 112.55 | **112.40** | ~0% |
| IGEV++ (RTIGEV) | 立体深度 | 143.40 | **143.06** | ~0% |

### 跟踪、分割、关键点、QR码检测

| 模型 | 任务 | 默认 (ms) | 优化后 (ms) | 提升 | FPS |
|------|------|-------:|--------:|-----:|----:|
| QR YOLO26n | QR码检测 | 4.08 | **3.63** | +12% | 275 |
| QR YOLO11n | QR码检测 | 4.26 | **3.80** | +12% | 263 |
| MixFormerV2 | 目标跟踪 | 10.75 | **10.42** | +3% | 96 |
| YOLOv7-Face | 人脸检测 | 12.93 | **12.66** | +2% | 79 |
| DeepLabv3Plus | 语义分割 | 13.83 | **13.24** | +4% | 76 |
| SuperPoint | 关键点 | 28.05 | **27.84** | +1% | 36 |
| DEIMv2 DINOv3-S | 检测 | 43.05 | **42.42** | +1% | 24 |

### 超分辨率、零样本、语音、修复

| 模型 | 任务 | 默认 (ms) | 优化后 (ms) | 提升 |
|------|------|-------:|--------:|-----:|
| Real-ESRGAN x4 | 64→256 放大 | 15.85 | **15.66** | +1% |
| Real-ESRGAN x4 | 256→1024 放大 | 476.2 | **475.4** | +0.2% |
| MobileSAM encoder | 分割 | 51.52 | **50.92** | +1% |
| MobileSAM decoder | 分割 | 10.59 | **10.35** | +2% |
| SigLIP2 vision | 零样本图像 | 11.48 | **11.37** | +1% |
| SigLIP2 text | 零样本文本 | 5.00 | **4.57** | +10% |
| RT-DETR | 检测 | 9.52 | **9.35** | +2% |
| YOLO-World | 检测 | 9.71 | **9.25** | +5% |
| Whisper-tiny enc | 语音识别 | 21.27 | **21.13** | +1% |
| Whisper-tiny dec | 语音识别 | 4.05 | **3.93** | +3% |
| RMBG-1.4 | 背景去除 | 107.2 | **106.5** | +1% |
| CodeFormer | 人脸修复 | 444.7 | **444.1** | +0.1% |
| DeOldify | 照片上色 | 383.6 | **383.0** | +0.2% |

### TTS — CosyVoice3（俄语文本，NPU 推理）

| 指标 | 默认 | 优化后 | 提升 |
|------|-----:|-------:|-----:|
| TTFT | 125 ms | **108 ms** | +16% |
| LLM 解码 | 13.9 tok/s | **16.3 tok/s** | +17% |
| RTF | 2.0-3.7x | **1.7-1.9x** | |

### 优化效果规律

加速效果与推理时间成反比 — 推理越快的模型受益越大：

| 推理时间 | 示例 | 提升 |
|:-:|:-:|:-:|
| ~0.27 ms | EdgeTAM prompt encoder | **+10%** |
| ~0.3 ms | Insightface genderage | **+34%** |
| < 0.5 ms | OCR 分类器 | **+71%** |
| ~0.7 ms | MobileNetV2 | **+50%** |
| ~1.4 ms | ResNet18, gtcrn | **+12-37%** |
| ~3.6 ms | QR YOLO26n/YOLO11n | +12% |
| ~3.7 ms | Insightface w600k_r50 | +15% |
| ~10 ms | MixFormerV2 | +3% |
| ~13 ms | DeepLabv3Plus | +4% |
| ~29 ms | OCR 检测器 | +1% |
| ~107 ms | RMBG-1.4 | +1% |
| ~143 ms | IGEV++ 立体深度 | ~0% |
| ~445 ms | CodeFormer | +0.1% |
| ~498 ms | DeOldify artistic | +0.2% |

这是因为 PCIe 往返延迟（约 0.3ms）在快速模型的总推理时间中占比更大。

**已测试 60+ 模型**，涵盖 21 个类别：LLM、目标检测、实例/语义分割、分类、OCR、人脸识别/修复、超分辨率、零样本、语音识别、TTS、立体深度、视频分割、说话人识别、音频降噪、目标跟踪、关键点检测、QR码检测、背景去除、照片上色等。

详情：[基准测试结果](docs/benchmark-results.md) | [PCIe 架构分析](docs/pcie-analysis.md)

## 系统要求

- FriendlyElec CM3588 NAS (RK3588)
- AX650N M.2 模块（[M5Stack Module LLM](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) 或同类产品）
- 已安装 AXCL 驱动（[ax-llm](https://github.com/AXERA-TECH/ax-llm)）
- Root 权限

## 卸载

```bash
sudo ./uninstall.sh
```

## 相关资源

- [CM3588 Wiki](https://wiki.friendlyelec.com/wiki/index.php/CM3588) — FriendlyElec CM3588 文档
- [M5Stack Module LLM](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) — M5Stack AI-8850 / AX650N 文档
- [ax-llm](https://github.com/AXERA-TECH/ax-llm) — 爱芯元智 LLM 推理引擎
- [AXCL](https://github.com/AXERA-TECH/axcl) — 爱芯元智 PCIe 主机 SDK

## 许可证

[MIT](LICENSE)
