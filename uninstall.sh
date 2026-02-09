#!/bin/bash
# Uninstall AX650N PCIe optimization
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}AX650N PCIe Optimization — Uninstaller${NC}"
echo

if [ "$(id -u)" -ne 0 ]; then
    echo -e "${RED}Error: must run as root${NC}"
    exit 1
fi

# Stop and disable service
if systemctl is-enabled ax650n-optimize.service &>/dev/null; then
    systemctl stop ax650n-optimize.service 2>/dev/null || true
    systemctl disable ax650n-optimize.service
    echo -e "${GREEN}[OK]${NC} Service disabled"
fi

# Remove files
rm -f /etc/systemd/system/ax650n-optimize.service
rm -f /usr/local/bin/ax650n-optimize.sh
rm -f /usr/local/bin/ax650n-diagnose.sh
systemctl daemon-reload

echo -e "${GREEN}[OK]${NC} Files removed"
echo
echo "Uninstall complete. Reboot to restore default settings."
