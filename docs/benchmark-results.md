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

## Methodology

- Each benchmark run: single prompt ("What is 2+2?" or similar short prompt)
- TTFT measured by ax-llm runtime
- Decode speed measured as average over full response generation
- "Cold" = first run after reboot (model loading from disk)
- "Warm" = subsequent runs (model in page cache)
