#!/bin/bash
# Install AX650N PCIe optimization for CM3588 NAS
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_PATH="/usr/local/bin/ax650n-optimize.sh"
DIAGNOSE_PATH="/usr/local/bin/ax650n-diagnose.sh"
SERVICE_PATH="/etc/systemd/system/ax650n-optimize.service"

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}AX650N PCIe Optimization — Installer${NC}"
echo

# Check root
if [ "$(id -u)" -ne 0 ]; then
    echo -e "${RED}Error: must run as root${NC}"
    echo "Usage: sudo ./install.sh"
    exit 1
fi

# Check RK3588
if ! grep -qi "rk3588\|rockchip" /proc/device-tree/compatible 2>/dev/null; then
    echo -e "${RED}Warning: this does not appear to be an RK3588 board${NC}"
    read -rp "Continue anyway? [y/N] " confirm
    [ "$confirm" != "y" ] && exit 1
fi

# Check AX650N
if ! lspci | grep -qi "axera\|0650"; then
    echo -e "${RED}Error: AX650N not detected on PCIe bus${NC}"
    echo "Check that M5Stack Module LLM is properly inserted in M.2 slot"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} AX650N detected: $(lspci | grep -i 'axera\|0650')"

# Install scripts
cp "$SCRIPT_DIR/scripts/ax650n-optimize.sh" "$INSTALL_PATH"
chmod +x "$INSTALL_PATH"
echo -e "${GREEN}[OK]${NC} Installed $INSTALL_PATH"

cp "$SCRIPT_DIR/scripts/ax650n-diagnose.sh" "$DIAGNOSE_PATH"
chmod +x "$DIAGNOSE_PATH"
echo -e "${GREEN}[OK]${NC} Installed $DIAGNOSE_PATH"

# Install systemd service
cp "$SCRIPT_DIR/systemd/ax650n-optimize.service" "$SERVICE_PATH"
systemctl daemon-reload
systemctl enable ax650n-optimize.service
echo -e "${GREEN}[OK]${NC} Service enabled (will start on boot)"

# Run optimization now
echo
echo "Applying optimization..."
"$INSTALL_PATH"

echo
echo -e "${GREEN}${BOLD}Installation complete!${NC}"
echo
echo "Commands available:"
echo "  ax650n-optimize.sh   — apply optimization (runs automatically on boot)"
echo "  ax650n-diagnose.sh   — full PCIe diagnostic report"
echo
echo "Service management:"
echo "  systemctl status ax650n-optimize"
echo "  journalctl -t ax650n-optimize"
