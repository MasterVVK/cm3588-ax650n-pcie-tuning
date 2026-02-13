# CM3588 + AX650N PCIe 性能优化

**[English](README.md)** | **[Русский](README.ru.md)**

> 在 [FriendlyElec CM3588 NAS](https://wiki.friendlyelec.com/wiki/index.php/CM3588) 上将 [M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) 的 LLM 推理速度提升 50-100%。

## 问题

AX650N NPU（24 TOPS INT8）通过 M.2 连接到 CM3588 NAS，LLM 推理速度仅有 **7-7.5 tok/s**（Qwen3-0.6B），低于预期的 12-13 tok/s。根本原因：

1. **PCIe Gen2 x1 硬件限制** — CM3588 仅为 M.2 插槽布线了 1 条通道（设备支持 x2）
2. **中断在小核上处理** — 所有中断由低速 Cortex-A55 @ 1.8 GHz 处理
3. **动态频率调节** — CPU 在推理调用间隙降低频率

## 解决方案

本工具包应用两项优化，总共带来 **+50-100% 性能提升**（取决于模型）：

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
| MiniCPM4-0.5B | W8A16 | 6-11 | **15-20** | +100% | 36 |
| SmolLM2-360M | W8A16 | 5-10 | **13-16** | +75% | 38.7 |
| Qwen3-0.6B | W8A16 | 7.1-7.5 | **10-12** | +50% | 19-20 |
| DeepSeek-R1-1.5B | W4A16 | 4.8-7.0 | **10.2-11.0** | +75% | 17.7 |
| DeepSeek-R1-1.5B | W8A16 | 4.0-5.2 | **7.6-8.6** | +95% | 17.7 |
| Qwen3-1.7B | W8A16 | 5.1-5.3 | **7.8-8.0** | +50% | 7.42 |
| Qwen2.5-7B | W4A16 | 3.7 | **4.4** | +19% | 4.8 |
| SmolLM3-3B | W8A16 | 2.6-3.2 | **4.3-4.4** | +50% | — |
| Qwen3-4B | W8A16 | 2.6-2.8 | **3.7** | +37% | — |

首字延迟 (TTFT)：

| 模型 | 默认 | 优化后 | 提升 |
|------|-----:|-------:|-----:|
| MiniCPM4-0.5B | 305-423 ms | **214-242 ms** | +40% |
| SmolLM2-360M | 348-530 ms | **259-304 ms** | +35% |
| Qwen3-0.6B | 488-578 ms | **391 ms** | +25% |
| DeepSeek-R1-1.5B | 503-688 ms | **380-432 ms** | +35% |
| Qwen3-1.7B | 541 ms | **447 ms** | +21% |
| SmolLM3-3B | 916-1043 ms | **708-735 ms** | +30% |
| Qwen3-4B | 1216 ms | **1110 ms** | +10% |

MiniCPM4-0.5B 和 SmolLM2-360M 展示了最高的优化增益 **+75-100%** — 小型高效架构从 PCIe 延迟降低中获益最大。未优化时基线极不稳定（schedutil 导致最多 2 倍波动）。DeepSeek-R1-1.5B W4A16 优化后达到 **11 tok/s**，使推理模型在边缘硬件上变得实用。Qwen3-1.7B 达到 **官方原生性能的约 108%**（7.9 vs 7.42）。**测试了 9 种 LLM 配置**，涵盖 7 个模型系列。

### 视觉模型（NPU 推理，640x640）

| 模型 | 默认 (ms) | 优化后 (ms) | 提升 | FPS |
|------|-------:|--------:|-----:|----:|
| YOLO11s | 3.99 | **3.55** | +12% | 282 |
| YOLOv8s | 4.21 | **3.89** | +8% | 257 |
| YOLO11s-Seg | 5.27 | **4.60** | +13% | 217 |
| YOLOv8s-Seg | 5.26 | **5.11** | +3% | 196 |
| YOLOv5s | 6.92 | **6.59** | +5% | 152 |
| YOLO26m | 9.47 | **9.04** | +5% | 111 |
| YOLOv5s-Seg | 10.08 | **9.86** | +2% | 101 |
| YOLOv8s-Pose | 11.81 | **11.26** | +5% | 89 |
| YOLO11x | 25.86 | **25.19** | +3% | 40 |
| YOLO11x-Pose | 25.65 | **25.55** | +0.4% | 39 |
| YOLO11x-Seg | 35.51 | **35.15** | +1% | 28 |
| Depth-Anything-V2-S | 34.00 | **33.44** | +2% | 30 |

### YOLO26 姿态估计 & 分割（NPU 3核，640x640）

| 模型 | 默认 (ms) | 优化后 (ms) | 提升 | FPS | Native (ms) |
|------|-------:|--------:|-----:|----:|------------:|
| YOLO26n-Pose | 2.08 | **1.71** | **+22%** | 586 | 1.53 |
| YOLO26n-Seg | 2.86 | **2.34** | **+22%** | 428 | 1.97 |
| YOLO26s-Pose | 4.07 | **3.72** | +10% | 269 | 3.53 |
| YOLO26s-Seg | 5.61 | **5.04** | +12% | 199 | 4.70 |
| YOLO26m-Pose | 10.25 | **9.62** | +7% | 104 | 9.30 |
| YOLO26x-Pose | 26.38 | **25.71** | +3% | 39 | 25.13 |

YOLO26 nano 模型：**+22%** — 视觉模型中最高加速。完整 n/s/m/l/x 结果见[基准测试详情](docs/benchmark-results.md)。

### Depth-Anything-3

| 模型 | 默认 (ms) | 优化后 (ms) | 提升 | Native (ms) |
|------|-------:|--------:|-----:|------------:|
| DA3-small | 24.02 | **23.28** | +3% | 22.77 |
| DA3-base | 68.51 | **67.71** | +1% | 67.34 |

视觉模型关键效果：**速度提升 2-22% + 延迟稳定性提高 2-3 倍**（对实时管线至关重要）。

### 分类模型（NPU 推理，224x224）

| 模型 | 默认 (ms) | 优化后 (ms) | 提升 | FPS |
|------|-------:|--------:|-----:|----:|
| MobileNetV2 | 0.983 | **0.657** | +50% | 1523 |
| SqueezeNet1.1 | 0.786 | **0.768** | +2% | 1302 |
| ResNet18 | 1.963 | **1.435** | +37% | 697 |
| ResNet50 | 3.613 | **3.355** | +8% | 298 |

### OCR — PPOCR_v5（文字检测 + 识别）

| 模型 | 任务 | 默认 (ms) | 优化后 (ms) | 提升 | Native (ms) |
|------|------|-------:|--------:|-----:|------------:|
| det_npu1 | 文字检测 | 29.38 | **28.98** | +1% | — |
| det_npu3 | 文字检测 | 18.41 | **17.75** | +4% | 16.8 |
| cls_npu1 | 文字方向 | 0.759 | **0.445** | +71% | — |
| cls_npu3 | 文字方向 | 0.429 | **0.351** | +18% | 0.17 |
| rec_npu1 | 文字识别 | 3.958 | **3.681** | +8% | — |
| rec_npu3 | 文字识别 | 2.104 | **1.719** | +18% | 1.4 |

npu3 为官方版本（使用全部 3 个 NPU 核心）。

### 场景文字识别 — SATRN

| 模型 | 默认 (ms) | 优化后 (ms) | 提升 | Native (ms) |
|------|-------:|--------:|-----:|------------:|
| SATRN backbone+encoder | 7.43 | **7.35** | +1% | 6.09 |
| SATRN decoder | 2.17 | **1.58** | **+37%** | 1.38 |

### 人脸识别 — Insightface

| 模型 | 任务 | 默认 (ms) | 优化后 (ms) | 提升 | FPS | Native (ms) |
|------|------|-------:|--------:|-----:|----:|------------:|
| det_10g | 人脸检测 | 7.36 | **6.86** | +7% | 146 | 6.95 |
| genderage | 性别/年龄 | 0.479 | **0.357** | +34% | 2801 | 0.30 |
| w600k_r50 | 特征提取 | 4.27 | **3.72** | +15% | 269 | 3.99 |

### 立体深度、视频分割、说话人识别、音频、肖像动画

| 模型 | 任务 | 默认 (ms) | 优化后 (ms) | 提升 | Native (ms) |
|------|------|-------:|--------:|-----:|------------:|
| LivePortrait stitching | 肖像动画 | 0.311 | **0.198** | **+57%** | — |
| EdgeTAM prompt enc | 视频分割 | 0.297 | **0.270** | +10% | 0.06 |
| EdgeTAM prompt mask | 视频分割 | 0.765 | **0.732** | +4% | 0.46 |
| gtcrn | 音频降噪 | 1.607 | **1.434** | +12% | — |
| 3D-Speaker ECAPA-TDNN | 说话人识别 | 4.006 | **3.889** | +3% | — |
| EdgeTAM mask decoder | 视频分割 | 5.338 | **5.184** | +3% | 4.73 |
| 3D-Speaker Res2NetV2 | 说话人识别 | 5.534 | **5.459** | +1% | 5.09 |
| LivePortrait motion | 肖像动画 | 8.177 | **7.472** | +9% | — |
| LivePortrait feature | 肖像动画 | 20.36 | **19.87** | +3% | — |
| RAFT-stereo 256x640 | 立体深度 | 21.19 | **21.28** | ~0% | 20.9 |
| EdgeTAM image encoder | 视频分割 | 23.88 | **23.73** | +1% | 22.35 |
| RAFT-stereo 384x1280 | 立体深度 | 112.55 | **112.40** | ~0% | — |
| IGEV++ (RTIGEV) | 立体深度 | 143.40 | **143.06** | ~0% | 139.80 |
| centerpoint | 3D LiDAR 检测 | 92.92 | **92.26** | +0.7% | 88.3 |
| bevformer | 3D BEV 检测 | 92.58 | **92.11** | +0.5% | 91.2 |
| MeloTTS decoder | TTS 解码器 | 92.6 | **92.0** | +0.7% | — |
| LivePortrait spade | 肖像动画 | 233.3 | **232.5** | +0.3% | — |
| mel_band_roformer | 音乐分离 | 426.3 | **425.6** | +0.2% | — |

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

| 模型 | 任务 | 默认 (ms) | 优化后 (ms) | 提升 | Native (ms) |
|------|------|-------:|--------:|-----:|------------:|
| Real-ESRGAN x4 | 64→256 放大 | 15.85 | **15.66** | +1% | 15.0 |
| Real-ESRGAN x4 | 256→1024 放大 | 476.2 | **475.4** | +0.2% | 440 |
| MobileSAM encoder | 分割 | 51.52 | **50.92** | +1% | 49.5 |
| MobileSAM decoder | 分割 | 10.59 | **10.35** | +2% | 9.93 |
| SigLIP2 vision | 零样本图像 | 11.48 | **11.37** | +1% | 11.1 |
| SigLIP2 text | 零样本文本 | 5.00 | **4.57** | +10% | 4.56 |
| RT-DETR | 检测 | 9.52 | **9.35** | +2% | — |
| YOLO-World | 检测 | 9.71 | **9.25** | +5% | — |
| Whisper-tiny enc | 语音识别 | 21.27 | **21.13** | +1% | — |
| Whisper-tiny dec | 语音识别 | 4.05 | **3.93** | +3% | — |
| Zipformer joiner | ASR | 0.664 | **0.344** | **+93%** | — |
| Zipformer decoder | ASR | 0.437 | **0.243** | **+80%** | — |
| Zipformer encoder | ASR | 3.602 | **3.018** | +19% | — |
| SenseVoice 流式 | ASR（5种语言） | 13.11 | **12.37** | +6% | — |
| SenseVoice 完整 | ASR（5种语言） | 55.33 | **54.69** | +1% | — |
| MobileCLIP2-S0 | CLIP 图像 | 8.63 | **8.49** | +2% | — |
| MobileCLIP2-S4 | CLIP 图像 | 64.94 | **64.34** | +1% | 65.3 |
| LibCLIP cnclip vision | CLIP 图像（中文） | 89.44 | **88.76** | +0.8% | 88.48 |
| LibCLIP cnclip text | CLIP 文本（中文） | 5.04 | **4.58** | +10% | 4.58 |
| FG-CLIP image | CLIP 图像 | 129.2 | **128.7** | +0.4% | 125.2 |
| FG-CLIP text | CLIP 文本 | 11.67 | **11.08** | +5% | 10.82 |
| jina-clip-v2 image | CLIP 图像 | 597.2 | **596.2** | +0.2% | 592.2 |
| jina-clip-v2 text | CLIP 文本 | 15.31 | **14.86** | +3% | 15.48 |
| ESPCN x2 | 超分辨率 | 22.71 | **22.41** | +1% | 22 |
| CLIP ViT-L/14 text | CLIP 文本 | 6.39 | **5.82** | +10% | — |
| CLIP ViT-L/14 image | CLIP 图像 | 71.11 | **70.28** | +1% | — |
| SigLIP-so400m vision | 零样本图像 | 168.9 | **168.2** | +0.5% | — |
| SigLIP-so400m text | 零样本文本 | 23.51 | **22.88** | +3% | — |
| RMBG-1.4 | 背景去除 | 107.2 | **106.5** | +1% | — |
| CodeFormer | 人脸修复 | 444.7 | **444.1** | +0.1% | — |
| DeOldify | 照片上色 | 383.6 | **383.0** | +0.2% | — |
| EDSR baseline x2 | 超分辨率 | 694.7 | **693.9** | +0.1% | — |

### TTS — CosyVoice3（俄语文本，NPU 推理）

| 指标 | 默认 | 优化后 | 提升 |
|------|-----:|-------:|-----:|
| TTFT | 125 ms | **108 ms** | +16% |
| LLM 解码 | 13.9 tok/s | **16.3 tok/s** | +17% |
| RTF | 2.0-3.7x | **1.7-1.9x** | |

### Embedding / RAG

| 模型 | 默认 (ms) | 优化后 (ms) | 提升 | Native (ms) |
|------|-------:|--------:|-----:|------------:|
| bge-small-en-v1.5 | 35.25 | **34.74** | +1.5% | 32.4 |
| Qwen3-Embedding Layer (×28) | 2.03 | **1.80** | +12.5% | — |
| Qwen3-Embedding Post | 8.67 | **8.09** | +7% | — |

Qwen3-Embedding（28层，W8A16）— 与 Qwen3-0.6B LLM 相同架构。解码层加速 +12.5%，与 LLM 模式一致。

### VLM — SmolVLM2-256M 和 FastVLM-0.5B（组件基准测试）

无 AXCL aarch64 二进制文件可用于端到端 VLM 推理；通过 `axcl_run_model` 单独测试各 .axmodel 组件：

| 模型 | 组件 | 默认 (ms) | 优化后 (ms) | 提升 |
|------|------|-------:|--------:|-----:|
| SmolVLM2-256M | 视觉编码器 | 99.12 | **98.35** | +1% |
| SmolVLM2-256M | LLM 层 | 0.818 | **0.566** | **+45%** |
| SmolVLM2-256M | LLM Post | 1.980 | **1.639** | +21% |
| FastVLM-0.5B | 视觉编码器 | 45.51 | **44.64** | +2% |
| FastVLM-0.5B | LLM 层 | 1.547 | **1.170** | **+32%** |
| FastVLM-0.5B | LLM Post | 7.538 | **7.043** | +7% |
| SmolVLM-256M | 视觉编码器 | 99.26 | **98.58** | +1% |
| SmolVLM-256M | LLM 层 | 0.700 | **0.450** | **+56%** |
| SmolVLM-256M | LLM Post | 2.110 | **1.582** | +33% |
| InternVL2.5-1B | 视觉编码器 | 357.8 | **357.0** | +0.2% |
| InternVL2.5-1B | LLM 层 | 1.648 | **1.020** | **+62%** |
| InternVL2.5-1B | LLM Post | 7.533 | **7.098** | +6% |

VLM 解码层加速 +32-62% — 与 LLM 模式一致。预估解码速度：SmolVLM2 ~38→54 tok/s (+42%，native: 76.7)，SmolVLM ~43→66 tok/s (+53%，native: 80)，FastVLM ~22→29 tok/s (+28%，native: 34.8)，InternVL2.5-1B ~21→32 tok/s (+52%，**达到 native 速度**：32)。

### 优化效果规律

加速效果与推理时间成反比 — 推理越快的模型受益越大：

| 推理时间 | 示例 | 提升 |
|:-:|:-:|:-:|
| ~0.20 ms | LivePortrait stitching | **+57%** |
| ~0.24 ms | Zipformer decoder | **+80%** |
| ~0.27 ms | EdgeTAM prompt encoder | **+10%** |
| ~0.3 ms | Insightface genderage | **+34%** |
| ~0.34 ms | Zipformer joiner | **+93%** |
| ~0.35 ms | PPOCR_v5 cls (npu3) | **+18%** |
| ~0.45 ms | SmolVLM-256M LLM 层 | **+56%** |
| < 0.5 ms | OCR 分类器 (npu1) | **+71%** |
| ~0.7 ms | MobileNetV2 | **+50%** |
| ~1.0 ms | InternVL2.5-1B LLM 层 | **+62%** |
| ~1.6 ms | SATRN decoder | **+37%** |
| ~1.7 ms | PPOCR_v5 rec (npu3) | **+18%** |
| ~1.4 ms | ResNet18, gtcrn | **+12-37%** |
| ~3.0 ms | Zipformer encoder | +19% |
| ~3.6 ms | QR YOLO26n/YOLO11n | +12% |
| ~3.7 ms | Insightface w600k_r50 | +15% |
| ~4.0 ms | YOLOv8s 检测 | +10% |
| ~5.0 ms | YOLOv8s-Seg | +4% |
| ~5.8 ms | CLIP ViT-L/14 文本 | +10% |
| ~7.5 ms | LivePortrait motion | +9% |
| ~10 ms | MixFormerV2/YOLOv5s-Seg | +2-3% |
| ~13 ms | DeepLabv3Plus | +4% |
| ~18 ms | PPOCR_v5 det (npu3) | +4% |
| ~20 ms | LivePortrait feature | +3% |
| ~25 ms | YOLO11x/YOLO11x-Pose | +0.4-3% |
| ~29 ms | OCR 检测器 | +1% |
| ~35 ms | YOLO11x-Seg | +1% |
| ~92 ms | centerpoint/bevformer/MeloTTS | +0.5-0.7% |
| ~107 ms | RMBG-1.4 | +1% |
| ~143 ms | IGEV++ 立体深度 | ~0% |
| ~233 ms | LivePortrait spade | +0.3% |
| ~445 ms | CodeFormer | +0.1% |
| ~358 ms | InternVL2.5-1B 视觉 | +0.2% |
| ~498 ms | DeOldify artistic | +0.2% |
| ~694 ms | EDSR baseline x2 | +0.1% |

这是因为 PCIe 往返延迟（约 0.3ms）在快速模型的总推理时间中占比更大。

Zipformer joiner (+93%) 是单次推理模型的最高加速记录。VLM 解码层持续显示 +32-62% 的提升。InternVL2.5-1B 优化后**达到官方原生速度**（32 tok/s）。

**已测试 130+ 模型**，涵盖 29 个类别：LLM、VLM、目标检测、姿态估计、实例/语义分割、分类、OCR、人脸识别/修复、超分辨率、零样本、CLIP、语音识别、TTS、深度估计、立体深度、视频分割、说话人识别、音频降噪、目标跟踪、关键点检测、QR码检测、背景去除、照片上色、肖像动画、3D 目标检测等。

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
