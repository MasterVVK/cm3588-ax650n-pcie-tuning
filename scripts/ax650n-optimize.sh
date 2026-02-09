#!/bin/bash
# AX650N PCIe Performance Optimization for CM3588 NAS
# Moves AX650N IRQ to big A76 core + sets performance CPU governor
#
# Effect: ~100% improvement in LLM inference speed (5-7 tok/s → 12+ tok/s)

set -e

LOG_TAG="ax650n-optimize"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { logger -t "$LOG_TAG" "$1"; echo -e "${GREEN}[OK]${NC} $1"; }
warn() { logger -t "$LOG_TAG" "WARN: $1"; echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { logger -t "$LOG_TAG" "ERROR: $1"; echo -e "${RED}[ERROR]${NC} $1"; }

# --- Auto-detect AX650N ---

find_ax650n() {
    local bdf
    bdf=$(lspci -D | grep -i "axera" | awk '{print $1}')
    if [ -z "$bdf" ]; then
        bdf=$(lspci -D | grep "0650" | awk '{print $1}')
    fi
    echo "$bdf"
}

# --- Auto-detect big cores (A76 on RK3588: CPU4-7) ---

find_big_cores() {
    local big_cores=""
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        local num
        num=$(basename "$cpu" | sed 's/cpu//')
        local max_freq
        max_freq=$(cat "$cpu/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo 0)
        # A76 cores on RK3588 run at 2.3+ GHz (2304000+ KHz)
        if [ "$max_freq" -ge 2200000 ]; then
            big_cores="$big_cores $num"
        fi
    done
    echo "$big_cores"
}

# --- Main ---

echo "AX650N PCIe Performance Optimizer"
echo "================================="
echo

# Find AX650N
PCIE_BDF=$(find_ax650n)
if [ -z "$PCIE_BDF" ]; then
    err "AX650N not found on PCIe bus"
    err "Run 'lspci | grep -i axera' to check"
    exit 1
fi
log "Found AX650N at $PCIE_BDF"

# Wait for device IRQ (may not be ready immediately after boot)
for i in $(seq 1 30); do
    if grep -q "$PCIE_BDF" /proc/interrupts 2>/dev/null; then
        break
    fi
    [ "$i" -eq 1 ] && echo -n "Waiting for device IRQ..."
    echo -n "."
    sleep 1
done
echo

# Find IRQ number
IRQ=$(grep "$PCIE_BDF" /proc/interrupts | awk '{print $1}' | tr -d ':')
if [ -z "$IRQ" ]; then
    err "Cannot find IRQ for $PCIE_BDF in /proc/interrupts"
    exit 1
fi

# Current state
CURRENT_AFF=$(cat "/proc/irq/$IRQ/smp_affinity" 2>/dev/null)
CURRENT_CPU=$(grep "$PCIE_BDF" /proc/interrupts | awk '{for(i=2;i<=NF;i++){if($i~/^[0-9]+$/ && $i>0){print i-2; exit}}}')
echo "Current state:"
echo "  IRQ $IRQ affinity mask: $CURRENT_AFF (CPU${CURRENT_CPU:-?})"

# Find big cores
BIG_CORES=$(find_big_cores)
if [ -z "$BIG_CORES" ]; then
    warn "Cannot detect big cores, defaulting to CPU4"
    BIG_CORES="4"
fi

# Use first big core
TARGET_CPU=$(echo "$BIG_CORES" | awk '{print $1}')
# CPU affinity mask: bit N = CPU N
MASK=$(printf '%x' $((1 << TARGET_CPU)))

# Apply IRQ affinity
echo "$MASK" > "/proc/irq/$IRQ/smp_affinity"
log "IRQ $IRQ affinity set to CPU${TARGET_CPU} (mask=0x${MASK})"

# Set performance governor for big cores
for cpu_num in $BIG_CORES; do
    gov_path="/sys/devices/system/cpu/cpu${cpu_num}/cpufreq/scaling_governor"
    if [ -f "$gov_path" ]; then
        PREV_GOV=$(cat "$gov_path")
        echo performance > "$gov_path"
    fi
done
log "CPU governor set to 'performance' for big cores (CPU${BIG_CORES// /, CPU})"

# Show PCIe link info
echo
echo "PCIe link status:"
LNKSTA=$(lspci -vvv -s "$PCIE_BDF" 2>/dev/null | grep "LnkSta:" | head -1)
LNKCAP=$(lspci -vvv -s "$PCIE_BDF" 2>/dev/null | grep "LnkCap:" | head -1)
if [ -n "$LNKSTA" ]; then
    echo "  $LNKCAP"
    echo "  $LNKSTA"
fi

# Verify
echo
echo "Applied state:"
ACTUAL_AFF=$(cat "/proc/irq/$IRQ/smp_affinity")
ACTUAL_GOV=$(cat "/sys/devices/system/cpu/cpu${TARGET_CPU}/cpufreq/scaling_governor")
echo "  IRQ $IRQ affinity: 0x${ACTUAL_AFF} → CPU${TARGET_CPU}"
echo "  CPU${TARGET_CPU} governor: $ACTUAL_GOV"
log "Optimization complete: IRQ=$IRQ→CPU${TARGET_CPU}, governor=$ACTUAL_GOV"
