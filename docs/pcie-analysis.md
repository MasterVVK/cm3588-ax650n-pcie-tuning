# PCIe Architecture Analysis: AX650N on CM3588 NAS

## RK3588 PCIe Controllers

The Rockchip RK3588 SoC has 5 PCIe controllers:

| Controller | Address | Type | Max Lanes | Max Speed |
|-----------|---------|------|-----------|-----------|
| pcie3x4 | 0xfe150000 | Synopsys DWC | 4 | Gen3 (8 GT/s) |
| pcie3x2 | 0xfe160000 | Synopsys DWC | 2 | Gen3 (8 GT/s) |
| pcie2x1l0 | 0xfe170000 | Synopsys DWC | 1 | Gen3 (8 GT/s)* |
| pcie2x1l1 | 0xfe180000 | Synopsys DWC | 1 | Gen3 (8 GT/s)* |
| pcie2x1l2 | 0xfe190000 | Synopsys DWC | 1 | Gen2 (5 GT/s) |

\* Despite the name `pcie2x1l*`, the CM3588 device tree sets `max-link-speed = <3>` (Gen3) for l0 and l1. Only l2 is Gen2.

## CM3588 NAS Board Layout

The FriendlyElec CM3588 NAS board splits **all** PCIe controllers into single x1 lanes:

```
pcie@fe150000 (pcie3x4, 1 lane) → M.2 NVMe Slot 1 (Gen3 x1)
pcie@fe160000 (pcie3x2, 1 lane) → M.2 NVMe Slot 2 (Gen3 x1)
pcie@fe170000 (pcie2x1l0)       → M.2 NVMe Slot 3 (Gen3 x1)
pcie@fe180000 (pcie2x1l1)       → M.2 Slot 4 / AX650N (Gen3 x1, negotiated Gen2) ← HERE
pcie@fe190000 (pcie2x1l2)       → RTL8125 Ethernet (Gen2 x1)
```

Device tree for all controllers: `num-lanes = <1>`.

## AX650N PCIe Capabilities

```
Device: Axera Semiconductor AX650N (Device ID: 0650)
LnkCap: Speed 5GT/s (Gen2), Width x2
LnkSta: Speed 5GT/s (Gen2), Width x1 (DOWNGRADED)

DevCap MaxPayload: 128 bytes (hardware maximum)
DevCtl MaxPayload: 128 bytes
DevCtl MaxReadReq: 512 bytes

ASPM: Not supported
ExtTag: Not supported
MSI: Enabled, Count=1/4
```

## Root Cause: x2 → x1 Downgrade

**Two hardware limitations:**

1. **Width**: Root port only provides 1 physical lane (x1), but AX650N wants 2 (x2)
2. **Speed**: Root port supports Gen3 (8GT/s), but AX650N only supports Gen2 (5GT/s)

```
Root Port LnkCap: Speed 8GT/s (Gen3), Width x1  ← only 1 lane available
AX650N   LnkCap: Speed 5GT/s (Gen2), Width x2  ← device wants 2 lanes, only Gen2

Negotiated: Speed 5GT/s (limited by AX650N), Width x1 (limited by root port)
```

Neither limitation can be fixed in software. The link runs at the lowest common denominator.

Effective bandwidth:
- **Current**: PCIe Gen2 x1 = ~500 MB/s
- **Theoretical max**: PCIe Gen2 x2 = ~1000 MB/s (requires x2 root port)
- **If AX650N supported Gen3**: PCIe Gen3 x1 = ~985 MB/s (root port already capable)

## MaxPayload Impact

AX650N hardware limit: MaxPayload = 128 bytes.

Each PCIe Transaction Layer Packet (TLP) carries at most 128 bytes of payload with ~20 bytes of header overhead. This means ~13% overhead on every transfer.

For comparison, typical NVMe drives support 256-512 byte MaxPayload.

## MaxReadReq Experiments

| MaxReadReq | LLM tok/s | Status |
|-----------|-----------|--------|
| 128 | — | **CRASH** (driver failure) |
| 256 | — | **CRASH** (driver failure) |
| **512** | **9.11** | **Stable (default)** |
| 4096 | 6.38 | Stable but 30% slower |

The AXCL driver (`ax_pcie_host_dev`) is hardcoded to work with MaxReadReq=512. Changing it causes data corruption and "input tensor not found" errors.

## IRQ Affinity Impact

RK3588 CPU topology:
- CPU0-3: Cortex-A55 (little) @ 1.8 GHz
- CPU4-7: Cortex-A76 (big) @ 2.3 GHz

By default, Linux routes AX650N MSI interrupts to CPU0 (A55). During LLM inference, each token generation triggers multiple PCIe DMA transfers, each requiring interrupt handling.

| IRQ Location | Governor | tok/s | TTFT |
|-------------|----------|-------|------|
| CPU0 (A55) | schedutil | 5-7.5 (unstable) | 440-616 ms |
| **CPU4 (A76)** | **performance** | **11-12.6 (stable)** | **353-397 ms** |

Moving IRQ to A76 core: **+60-140% decode speed, +14-43% prefill speed**.

## Why RPi5 Is Faster at Same Gen2 x1

RPi5 achieves ~13 tok/s vs CM3588's 12.2 tok/s (optimized):

1. **BCM2712 vs DWC PCIe controller**: Broadcom's implementation may have lower DMA latency
2. **Single-core architecture**: RPi5's Cortex-A76 cores are all identical, no big.LITTLE scheduling overhead
3. **Remaining gap is small**: ~6% after optimization

## Why 24 TOPS ≠ LLM Speed

AX650N is rated at 24 TOPS (INT8). This is **compute throughput**, not memory bandwidth.

LLM inference (especially decode phase) is **memory-bandwidth bound**:
- Each token requires reading all model weights
- AX650N has ~7 GB DDR4 with ~25-30 GB/s bandwidth
- Qwen3-0.6B (W8A16): ~600 MB model → theoretical max ~42 tok/s from memory bandwidth alone
- Actual native (no PCIe): 19-20 tok/s (NPU overhead)
- Through PCIe: limited by transfer latency per token

The 24 TOPS compute capacity is utilized less than 50% during LLM inference. The bottleneck chain is:

```
AX650N Memory BW (~25 GB/s) → limits native to ~20 tok/s
          ↓
PCIe Gen2 x1 latency → limits CM3588 to ~12 tok/s
          ↓
IRQ on little core → degrades to ~5-7 tok/s (without optimization)
```
