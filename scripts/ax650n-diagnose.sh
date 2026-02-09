#!/bin/bash
# AX650N PCIe Diagnostic Tool for CM3588 NAS
# Collects PCIe configuration, performance bottlenecks, and recommendations

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

section() { echo -e "\n${BOLD}${CYAN}=== $1 ===${NC}"; }
good()    { echo -e "  ${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "  ${YELLOW}[!!]${NC} $1"; }
bad()     { echo -e "  ${RED}[!!]${NC} $1"; }
info()    { echo -e "  $1"; }

echo -e "${BOLD}AX650N PCIe Diagnostic Report${NC}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Kernel: $(uname -r)"

# --- System ---

section "System Info"
if [ -f /proc/device-tree/model ]; then
    info "Board: $(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')"
fi
info "CPU: $(lscpu | grep 'Model name' | head -1 | sed 's/.*:\s*//')"
info "Memory: $(free -h | awk '/Mem/{print $2}')"
info "Uptime: $(uptime -p)"

# --- Find AX650N ---

section "AX650N Detection"
BDF=$(lspci -D | grep -i "axera" | awk '{print $1}')
if [ -z "$BDF" ]; then
    BDF=$(lspci -D | grep "0650" | awk '{print $1}')
fi

if [ -z "$BDF" ]; then
    bad "AX650N NOT FOUND on PCIe bus!"
    echo "  Check physical connection of M.2 module"
    exit 1
fi
good "AX650N found at $BDF"

# Get root port BDF
ROOT_BUS=$(echo "$BDF" | cut -d: -f1-2)
ROOT_BDF=$(lspci -D -s "${ROOT_BUS}:00.0" 2>/dev/null | awk '{print $1}' | head -1)
if [ -z "$ROOT_BDF" ]; then
    # Try parent bus
    PARENT_BUS=$(echo "$BDF" | awk -F: '{printf "%s:%02x:00.0", $1, strtonum("0x"$2)-1}')
    ROOT_BDF=$(lspci -D -s "$PARENT_BUS" 2>/dev/null | awk '{print $1}' | head -1)
fi

# --- PCIe Topology ---

section "PCIe Topology"
lspci -tv 2>/dev/null | head -20
echo

# --- AX650N PCIe Configuration ---

section "AX650N PCIe Link"

LNKCAP=$(lspci -vvv -s "$BDF" 2>/dev/null | grep "LnkCap:" | head -1)
LNKSTA=$(lspci -vvv -s "$BDF" 2>/dev/null | grep "LnkSta:" | head -1)
DEVCAP=$(lspci -vvv -s "$BDF" 2>/dev/null | grep "DevCap:" | head -1)
DEVCTL=$(lspci -vvv -s "$BDF" 2>/dev/null | grep "MaxPayload\|MaxReadReq" | head -1)

info "Device: $BDF"
info "$LNKCAP"
info "$LNKSTA"
info "$DEVCTL"

# Parse link width
CAP_WIDTH=$(echo "$LNKCAP" | grep -oP 'Width x\K[0-9]+')
STA_WIDTH=$(echo "$LNKSTA" | grep -oP 'Width x\K[0-9]+')
CAP_SPEED=$(echo "$LNKCAP" | grep -oP 'Speed \K[0-9.]+')
STA_SPEED=$(echo "$LNKSTA" | grep -oP 'Speed \K[0-9.]+')

if [ -n "$CAP_WIDTH" ] && [ -n "$STA_WIDTH" ]; then
    if [ "$STA_WIDTH" -lt "$CAP_WIDTH" ]; then
        bad "Link width DOWNGRADED: x${CAP_WIDTH} → x${STA_WIDTH}"
    else
        good "Link width: x${STA_WIDTH} (matches capability)"
    fi
fi

if [ -n "$STA_SPEED" ]; then
    case "$STA_SPEED" in
        2.5) info "Link speed: Gen1 (2.5 GT/s) = ~250 MB/s per lane" ;;
        5*)  info "Link speed: Gen2 (5 GT/s) = ~500 MB/s per lane" ;;
        8*)  info "Link speed: Gen3 (8 GT/s) = ~1000 MB/s per lane" ;;
    esac
fi

# Effective bandwidth
if [ -n "$STA_SPEED" ] && [ -n "$STA_WIDTH" ]; then
    case "$STA_SPEED" in
        2.5) BW_PER_LANE=250 ;;
        5*)  BW_PER_LANE=500 ;;
        8*)  BW_PER_LANE=1000 ;;
        *)   BW_PER_LANE=0 ;;
    esac
    EFF_BW=$((BW_PER_LANE * STA_WIDTH))
    info "Effective bandwidth: ~${EFF_BW} MB/s"
fi

# MaxPayload analysis
MAX_PAYLOAD=$(echo "$DEVCTL" | grep -oP 'MaxPayload \K[0-9]+')
if [ -n "$MAX_PAYLOAD" ]; then
    if [ "$MAX_PAYLOAD" -le 128 ]; then
        warn "MaxPayload: ${MAX_PAYLOAD} bytes (small, higher PCIe overhead)"
    else
        good "MaxPayload: ${MAX_PAYLOAD} bytes"
    fi
fi

# --- Root Port ---

section "Root Port Configuration"
if [ -n "$ROOT_BDF" ]; then
    info "Root port: $ROOT_BDF"
    RP_LNKCAP=$(lspci -vvv -s "$ROOT_BDF" 2>/dev/null | grep "LnkCap:" | head -1)
    RP_WIDTH=$(echo "$RP_LNKCAP" | grep -oP 'Width x\K[0-9]+')
    info "$RP_LNKCAP"
    if [ -n "$RP_WIDTH" ] && [ "$RP_WIDTH" -eq 1 ]; then
        warn "Root port supports only x1 — hardware limitation of this M.2 slot"
    fi

    # Device tree info
    DT_NODE=$(lspci -vvv -s "$BDF" 2>/dev/null | grep "Device tree node" | awk '{print $NF}')
    if [ -n "$DT_NODE" ]; then
        PCIE_CTRL=$(echo "$DT_NODE" | grep -oP 'pcie@[a-f0-9]+')
        if [ -n "$PCIE_CTRL" ]; then
            info "PCIe controller: $PCIE_CTRL"
        fi
    fi
fi

# --- IRQ Configuration ---

section "IRQ Configuration"
IRQ_LINE=$(grep "$BDF" /proc/interrupts 2>/dev/null)
if [ -n "$IRQ_LINE" ]; then
    IRQ=$(echo "$IRQ_LINE" | awk '{print $1}' | tr -d ':')
    AFF=$(cat "/proc/irq/$IRQ/smp_affinity" 2>/dev/null)

    # Find which CPU handles most interrupts
    COUNTS=$(echo "$IRQ_LINE" | awk '{for(i=2;i<=NF;i++){if($i~/^[0-9]+$/){printf "%s:%s ", i-2, $i}}}')
    MAX_CPU=""
    MAX_COUNT=0
    for pair in $COUNTS; do
        cpu=${pair%%:*}
        cnt=${pair##*:}
        if [ "$cnt" -gt "$MAX_COUNT" ]; then
            MAX_COUNT=$cnt
            MAX_CPU=$cpu
        fi
    done

    info "IRQ: $IRQ, affinity mask: 0x${AFF}"
    info "Most interrupts on: CPU${MAX_CPU} (${MAX_COUNT} total)"

    # Check if on big or little core
    if [ -n "$MAX_CPU" ]; then
        MAX_FREQ=$(cat "/sys/devices/system/cpu/cpu${MAX_CPU}/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo 0)
        if [ "$MAX_FREQ" -lt 2200000 ]; then
            bad "IRQ handled by LITTLE core (CPU${MAX_CPU}, ${MAX_FREQ}KHz max)"
            warn "Move to big core for +30-100% performance improvement"
        else
            good "IRQ handled by big core (CPU${MAX_CPU}, ${MAX_FREQ}KHz max)"
        fi
    fi
fi

# --- CPU Governor ---

section "CPU Governor"
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    num=$(basename "$cpu" | sed 's/cpu//')
    gov=$(cat "$cpu/cpufreq/scaling_governor" 2>/dev/null || echo "N/A")
    freq=$(cat "$cpu/cpufreq/scaling_cur_freq" 2>/dev/null || echo "0")
    max_freq=$(cat "$cpu/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo "0")
    freq_mhz=$((freq / 1000))
    max_mhz=$((max_freq / 1000))

    if [ "$max_freq" -ge 2200000 ]; then
        core_type="A76"
        if [ "$gov" != "performance" ]; then
            warn "CPU${num} ($core_type): $gov @ ${freq_mhz}/${max_mhz}MHz — consider 'performance'"
        else
            good "CPU${num} ($core_type): $gov @ ${freq_mhz}/${max_mhz}MHz"
        fi
    else
        core_type="A55"
        info "CPU${num} ($core_type): $gov @ ${freq_mhz}/${max_mhz}MHz"
    fi
done

# --- AXCL Driver ---

section "AXCL Driver"
DRIVER=$(lspci -s "$BDF" -k 2>/dev/null | grep "driver" | awk '{print $NF}')
if [ -n "$DRIVER" ]; then
    good "Driver: $DRIVER"
else
    bad "No kernel driver loaded for AX650N"
fi

if command -v /usr/bin/axcl/axcl-smi &>/dev/null; then
    info "axcl-smi output:"
    timeout 10 /usr/bin/axcl/axcl-smi 2>/dev/null | while IFS= read -r line; do
        echo "    $line"
    done
fi

# --- AER Errors ---

section "PCIe Errors (AER)"
AER_PATH="/sys/bus/pci/devices/${BDF}/aer_dev_correctable"
if [ -f "$AER_PATH" ]; then
    ERRS=$(cat "$AER_PATH" 2>/dev/null | grep -v "^$" | grep -v " 0$" | head -5)
    if [ -z "$ERRS" ]; then
        good "No correctable errors"
    else
        warn "Correctable errors detected:"
        echo "$ERRS" | while read -r line; do info "  $line"; done
    fi
fi

# --- Recommendations ---

section "Recommendations"
ISSUES=0

if [ -n "$STA_WIDTH" ] && [ -n "$CAP_WIDTH" ] && [ "$STA_WIDTH" -lt "$CAP_WIDTH" ]; then
    warn "PCIe x${STA_WIDTH} (device supports x${CAP_WIDTH}): hardware limitation of this M.2 slot"
    warn "  → For full x${CAP_WIDTH} bandwidth, use a board with PCIe x${CAP_WIDTH}+ slot"
    ISSUES=$((ISSUES + 1))
fi

if [ -n "$MAX_CPU" ]; then
    MAX_FREQ=$(cat "/sys/devices/system/cpu/cpu${MAX_CPU}/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo 0)
    if [ "$MAX_FREQ" -lt 2200000 ]; then
        warn "IRQ on little core → run: ax650n-optimize.sh"
        ISSUES=$((ISSUES + 1))
    fi
fi

BIG_GOV=$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor 2>/dev/null)
if [ "$BIG_GOV" != "performance" ]; then
    warn "Big cores not at max frequency → run: ax650n-optimize.sh"
    ISSUES=$((ISSUES + 1))
fi

if ! systemctl is-active ax650n-optimize.service &>/dev/null; then
    warn "ax650n-optimize.service not active → run: install.sh"
    ISSUES=$((ISSUES + 1))
fi

if [ "$ISSUES" -eq 0 ]; then
    good "All optimizations applied!"
fi

echo
echo "Done. Report generated at $(date)."
