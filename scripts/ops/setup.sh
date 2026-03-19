#!/bin/bash
# Aster Quick Setup - One-command installation and configuration

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/scripts/ops"
VENV_PATH="$PROJECT_ROOT/.venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Aster Background Service Setup                     ║"
echo "║    Local AI Inference Engine for macOS                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}✗ Virtual environment not found at $VENV_PATH${NC}"
    echo "Please run: python -m venv .venv && source .venv/bin/activate && pip install -e ."
    exit 1
fi

# Check if Python is available
if ! command -v "$VENV_PATH/bin/python" &> /dev/null; then
    echo -e "${RED}✗ Python not found in virtual environment${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Virtual environment found${NC}"
echo ""

# Menu
echo "What would you like to do?"
echo ""
echo "1. Install as background service (auto-start on boot)"
echo "2. Start the service"
echo "3. Stop the service"
echo "4. Check service status"
echo "5. View service logs"
echo "6. Enable/disable individual services (ASR, TTS)"
echo "7. Run health check"
echo "8. Run tests"
echo "9. Uninstall service"
echo "0. Exit"
echo ""
read -p "Enter your choice (0-9): " choice

case $choice in
    1)
        echo ""
        echo -e "${YELLOW}Installing Aster as background service...${NC}"
        "$VENV_PATH/bin/python" "$SCRIPTS_DIR/daemon.py" install
        echo ""
        echo -e "${GREEN}✓ Installation complete!${NC}"
        echo "The service will auto-start on next boot."
        echo "To start now, run: aster daemon start"
        ;;
    2)
        echo ""
        echo -e "${YELLOW}Starting Aster service...${NC}"
        "$VENV_PATH/bin/python" "$SCRIPTS_DIR/daemon.py" start
        ;;
    3)
        echo ""
        echo -e "${YELLOW}Stopping Aster service...${NC}"
        "$VENV_PATH/bin/python" "$SCRIPTS_DIR/daemon.py" stop
        ;;
    4)
        echo ""
        "$VENV_PATH/bin/python" "$SCRIPTS_DIR/daemon.py" status
        ;;
    5)
        echo ""
        echo -e "${YELLOW}Viewing service logs (press Ctrl+C to exit)...${NC}"
        echo ""
        "$VENV_PATH/bin/python" "$SCRIPTS_DIR/daemon.py" logs
        ;;
    6)
        echo ""
        echo "Service Management:"
        echo ""
        echo "1. List services"
        echo "2. Enable ASR"
        echo "3. Disable ASR"
        echo "4. Enable TTS"
        echo "5. Disable TTS"
        echo "6. Show service status"
        echo "0. Back"
        echo ""
        read -p "Enter your choice (0-6): " service_choice
        
        case $service_choice in
            1)
                "$VENV_PATH/bin/python" "$SCRIPTS_DIR/service_manager.py" list
                ;;
            2)
                "$VENV_PATH/bin/python" "$SCRIPTS_DIR/service_manager.py" enable ASR
                ;;
            3)
                "$VENV_PATH/bin/python" "$SCRIPTS_DIR/service_manager.py" disable ASR
                ;;
            4)
                "$VENV_PATH/bin/python" "$SCRIPTS_DIR/service_manager.py" enable TTS
                ;;
            5)
                "$VENV_PATH/bin/python" "$SCRIPTS_DIR/service_manager.py" disable TTS
                ;;
            6)
                "$VENV_PATH/bin/python" "$SCRIPTS_DIR/service_manager.py" status
                ;;
            0)
                ;;
            *)
                echo -e "${RED}Invalid choice${NC}"
                ;;
        esac
        ;;
    7)
        echo ""
        echo -e "${YELLOW}Running health check...${NC}"
        "$VENV_PATH/bin/python" "$SCRIPTS_DIR/health_monitor.py" check
        ;;
    8)
        echo ""
        echo -e "${YELLOW}Running tests...${NC}"
        "$VENV_PATH/bin/python" "$SCRIPTS_DIR/aster" test
        ;;
    9)
        echo ""
        read -p "Are you sure you want to uninstall? (y/N): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo -e "${YELLOW}Uninstalling Aster service...${NC}"
            "$VENV_PATH/bin/python" "$SCRIPTS_DIR/daemon.py" uninstall
            echo -e "${GREEN}✓ Service uninstalled${NC}"
        else
            echo "Cancelled"
        fi
        ;;
    0)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
