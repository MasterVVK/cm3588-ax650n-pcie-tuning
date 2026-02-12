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
| Qwen3-0.6B | W8A16 | 7.1-7.5 | **10-12** | +50% | 19-20 |
| Qwen3-1.7B | W8A16 | 5.1-5.3 | **7.8-8.0** | +50% | 7.42 |
| Qwen3-4B | W8A16 | 2.6-2.8 | **3.7** | +37% | — |
| Qwen2.5-7B | W4A16 | 3.7 | **4.4** | +19% | 4.8 |

首字延迟 (TTFT)：

| 模型 | 默认 | 优化后 | 提升 |
|------|-----:|-------:|-----:|
| Qwen3-0.6B | 488-578 ms | **391 ms** | +25% |
| Qwen3-1.7B | 541 ms | **447 ms** | +21% |
| Qwen3-4B | 1216 ms | **1110 ms** | +10% |

Qwen3-1.7B 优化后达到 **官方原生性能的约 108%**（7.9 vs 7.42）— 可能是测量误差，但 PCIe 开销对计算密集型模型实际上已被消除。Qwen2.5-7B：**原生的 92%**（4.4 vs 4.8）。

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

### 超分辨率、分割、零样本、语音

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
| ~0.3 ms | Insightface genderage | **+34%** |
| < 0.5 ms | OCR 分类器 | **+71%** |
| ~0.7 ms | MobileNetV2 | **+50%** |
| ~1.4 ms | ResNet18 | **+37%** |
| ~3.7 ms | Insightface w600k_r50 | +15% |
| ~3.5 ms | ResNet50 | +8% |
| ~7 ms | YOLOv5s | +5% |
| ~29 ms | OCR 检测器 | +1% |
| ~475 ms | Real-ESRGAN 256→1024 | +0.2% |

这是因为 PCIe 往返延迟（约 0.3ms）在快速模型的总推理时间中占比更大。

**已测试 30+ 模型**，涵盖 10 个类别：LLM（4 种模型大小，0.6B 至 7B）、视觉检测、分割、分类、OCR、人脸识别、超分辨率、零样本、语音识别和 TTS。

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
