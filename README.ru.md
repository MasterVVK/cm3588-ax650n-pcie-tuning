# CM3588 + AX650N — Оптимизация PCIe

**[English](README.md)** | **[中文](README.zh.md)**

> Удвоение скорости LLM inference для [M5Stack Module LLM (AI-8850)](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) на [FriendlyElec CM3588 NAS](https://wiki.friendlyelec.com/wiki/index.php/CM3588).

## Проблема

NPU AX650N (24 TOPS INT8), подключённый через M.2 на CM3588 NAS, выдаёт только **5-7 tok/s** при LLM inference вместо ожидаемых 12-13 tok/s. Причины:

1. **Аппаратное ограничение PCIe Gen2 x1** — CM3588 разводит только 1 линию к M.2 слоту (устройство поддерживает x2)
2. **IRQ на маленьком ядре** — все прерывания обрабатываются медленным Cortex-A55 @ 1.8 ГГц
3. **Динамическое масштабирование частоты** — CPU снижает частоту между вызовами inference

## Решение

Этот toolkit применяет две оптимизации, дающие **+100% прирост**:

| | До | После |
|--|-----|-------|
| Скорость декода | 5-7.5 tok/s | **11-12.6 tok/s** |
| TTFT (prefill) | 440-616 мс | **353-397 мс** |
| Стабильность | Высокий разброс | Стабильно |

## Быстрый старт

```bash
git clone https://github.com/MasterVVK/cm3588-ax650n-pcie-tuning.git
cd cm3588-ax650n-pcie-tuning
sudo ./install.sh
```

Оптимизация сохраняется после перезагрузок через systemd service.

## Что делает

1. **Переносит IRQ AX650N на большое ядро** — с Cortex-A55 (CPU0) на Cortex-A76 (CPU4)
2. **Устанавливает performance governor** — фиксирует частоту больших ядер на 2.3 ГГц
3. **Автоопределение** PCIe адреса AX650N и топологии CPU
4. **Сохраняется после ребута** через systemd service

## Диагностика

```bash
sudo ax650n-diagnose.sh
```

Показывает: PCIe топологию, скорость/ширину линка, MaxPayload, привязку IRQ, governor CPU и рекомендации.

## Результаты бенчмарков

### LLM Inference

Модель: Qwen3-0.6B (W8A16) через AXCL runtime

| Платформа | PCIe | tok/s |
|-----------|------|-------|
| AX650N native | — | 19-20 |
| Raspberry Pi 5 | Gen2 x1 | ~13 |
| **CM3588 (оптимизировано)** | **Gen2 x1** | **12.0** |
| CM3588 (по умолчанию) | Gen2 x1 | 5-7 |

### Vision модели (NPU inference, 640x640)

| Модель | По умолч. (мс) | Оптимизировано (мс) | Ускорение | FPS |
|--------|----------:|------------:|--------:|----:|
| YOLO11s | 3.99 | **3.55** | +12% | 282 |
| YOLOv8s | 4.21 | **3.89** | +8% | 257 |
| YOLO11s-Seg | 5.27 | **4.60** | +13% | 217 |
| YOLOv8s-Seg | 5.26 | **5.11** | +3% | 196 |
| YOLOv5s | 6.92 | **6.59** | +5% | 152 |
| YOLO26m | 9.47 | **9.04** | +5% | 111 |
| YOLOv8s-Pose | 11.81 | **11.26** | +5% | 89 |
| Depth-Anything-V2-S | 34.00 | **33.44** | +2% | 30 |

Ключевой эффект для vision: **2-13% быстрее + стабильность в 2-3 раза выше** (критично для realtime pipeline).

Подробнее: [результаты бенчмарков](docs/benchmark-results.md) и [анализ PCIe архитектуры](docs/pcie-analysis.md).

## Требования

- FriendlyElec CM3588 NAS (RK3588)
- AX650N M.2 модуль ([M5Stack Module LLM](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) или аналог)
- Установленный драйвер AXCL ([ax-llm](https://github.com/AXERA-TECH/ax-llm))
- Root-доступ

## Удаление

```bash
sudo ./uninstall.sh
```

## Ссылки

- [CM3588 Wiki](https://wiki.friendlyelec.com/wiki/index.php/CM3588) — документация FriendlyElec CM3588
- [M5Stack Module LLM](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) — документация M5Stack AI-8850 / AX650N
- [ax-llm](https://github.com/AXERA-TECH/ax-llm) — движок LLM inference от Axera
- [AXCL](https://github.com/AXERA-TECH/axcl) — Axera PCIe host SDK

## Лицензия

[MIT](LICENSE)
