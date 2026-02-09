# CM3588 + AX650N PCIe 性能优化

**[English](README.md)** | **[Русский](README.ru.md)**

> 在 [FriendlyElec CM3588 NAS](https://wiki.friendlyelec.com/wiki/index.php/CM3588) 上将 [M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) 的 LLM 推理速度提升一倍。

## 问题

AX650N NPU（24 TOPS INT8）通过 M.2 连接到 CM3588 NAS，LLM 推理速度仅有 **5-7 tok/s**，远低于预期的 12-13 tok/s。根本原因：

1. **PCIe Gen2 x1 硬件限制** — CM3588 仅为 M.2 插槽布线了 1 条通道（设备支持 x2）
2. **中断在小核上处理** — 所有中断由低速 Cortex-A55 @ 1.8 GHz 处理
3. **动态频率调节** — CPU 在推理调用间隙降低频率

## 解决方案

本工具包应用两项优化，总共带来 **+100% 性能提升**：

| | 优化前 | 优化后 |
|--|--------|--------|
| 解码速度 | 5-7.5 tok/s | **11-12.6 tok/s** |
| 首字延迟 (TTFT) | 440-616 ms | **353-397 ms** |
| 稳定性 | 波动大 | 稳定 |

## 快速开始

```bash
git clone https://github.com/YOUR_USER/cm3588-ax650n-pcie-tuning.git
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

模型：Qwen3-0.6B (W8A16)，AXCL 运行时

| 平台 | PCIe | tok/s |
|------|------|-------|
| AX650N 原生 | — | 19-20 |
| Raspberry Pi 5 | Gen2 x1 | ~13 |
| **CM3588（已优化）** | **Gen2 x1** | **12.0** |
| CM3588（默认） | Gen2 x1 | 5-7 |

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
