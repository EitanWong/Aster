#!/usr/bin/env python3
"""
Aster Daemon Manager - Control background service on macOS

Usage:
    python daemon.py install      # Install as launchd service
    python daemon.py uninstall    # Remove launchd service
    python daemon.py start        # Start the service
    python daemon.py stop         # Stop the service
    python daemon.py restart      # Restart the service
    python daemon.py status       # Check service status
    python daemon.py logs         # Tail service logs
    python daemon.py enable       # Enable auto-start on boot
    python daemon.py disable      # Disable auto-start on boot
    python daemon.py config       # Show configuration
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
VENV_PATH = PROJECT_ROOT / ".venv"
PYTHON_BIN = VENV_PATH / "bin" / "python"
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"
LOG_DIR = PROJECT_ROOT / "logs"
PID_FILE = LOG_DIR / "aster.pid"
LOG_FILE = LOG_DIR / "aster.log"
ERROR_LOG = LOG_DIR / "aster.error.log"

# macOS LaunchD configuration
LAUNCHD_LABEL = "com.local.aster.daemon"
LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"

# Service configuration
SERVICE_CONFIG = {
    "name": "Aster Inference Engine",
    "description": "Local AI inference service (LLM, ASR, TTS)",
    "port": 8080,
    "host": "127.0.0.1",
    "auto_restart": True,
    "restart_delay": 5,
    "max_restarts": 10,
    "health_check_interval": 30,
}


def ensure_log_dir() -> None:
    """Ensure log directory exists."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_plist_content() -> str:
    """Generate launchd plist content."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{PYTHON_BIN}</string>
        <string>-m</string>
        <string>aster</string>
        <string>--config</string>
        <string>{CONFIG_FILE}</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>{PROJECT_ROOT}</string>
    
    <key>StandardOutPath</key>
    <string>{LOG_FILE}</string>
    
    <key>StandardErrorPath</key>
    <string>{ERROR_LOG}</string>
    
    <key>KeepAlive</key>
    <true/>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>StartInterval</key>
    <integer>60</integer>
    
    <key>ProcessType</key>
    <string>Background</string>
    
    <key>Nice</key>
    <integer>10</integer>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{VENV_PATH}/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
</dict>
</plist>
"""


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run shell command."""
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(f"  Error: {e.stderr}", file=sys.stderr)
        raise


def install() -> int:
    """Install as launchd service."""
    print("Installing Aster as macOS background service...")
    
    ensure_log_dir()
    
    # Create plist file
    LAUNCHD_PLIST.parent.mkdir(parents=True, exist_ok=True)
    LAUNCHD_PLIST.write_text(get_plist_content())
    print(f"✓ Created launchd plist: {LAUNCHD_PLIST}")
    
    # Load service
    try:
        run_command(["launchctl", "load", str(LAUNCHD_PLIST)])
        print(f"✓ Service installed and loaded")
        print(f"✓ Service will auto-start on next boot")
        return 0
    except subprocess.CalledProcessError:
        print("✗ Failed to load service")
        return 1


def uninstall() -> int:
    """Remove launchd service."""
    print("Uninstalling Aster background service...")
    
    if not LAUNCHD_PLIST.exists():
        print("✗ Service not installed")
        return 1
    
    try:
        run_command(["launchctl", "unload", str(LAUNCHD_PLIST)])
        LAUNCHD_PLIST.unlink()
        print(f"✓ Service uninstalled")
        return 0
    except subprocess.CalledProcessError:
        print("✗ Failed to unload service")
        return 1


def start() -> int:
    """Start the service."""
    print("Starting Aster service...")
    
    try:
        run_command(["launchctl", "start", LAUNCHD_LABEL])
        print(f"✓ Service started")
        time.sleep(2)
        return status()
    except subprocess.CalledProcessError:
        print("✗ Failed to start service")
        return 1


def stop() -> int:
    """Stop the service."""
    print("Stopping Aster service...")
    
    try:
        run_command(["launchctl", "stop", LAUNCHD_LABEL], check=False)
        print(f"✓ Service stopped")
        return 0
    except subprocess.CalledProcessError:
        print("✗ Failed to stop service")
        return 1


def restart() -> int:
    """Restart the service."""
    print("Restarting Aster service...")
    stop()
    time.sleep(1)
    return start()


def status() -> int:
    """Check service status."""
    try:
        result = run_command(["launchctl", "list", LAUNCHD_LABEL], check=False)
        
        if result.returncode == 0:
            print("✓ Service is running")
            
            # Try to check API health
            try:
                import httpx
                response = httpx.get("http://127.0.0.1:8080/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    print(f"  API Status: {data.get('status', 'unknown')}")
                    print(f"  Degraded: {data.get('degraded', False)}")
            except Exception:
                print("  API: Not responding (may be starting)")
            
            return 0
        else:
            print("✗ Service is not running")
            return 1
    except subprocess.CalledProcessError:
        print("✗ Service is not installed")
        return 1


def logs() -> int:
    """Tail service logs."""
    if not LOG_FILE.exists():
        print("✗ No logs found")
        return 1
    
    try:
        subprocess.run(["tail", "-f", str(LOG_FILE)])
        return 0
    except KeyboardInterrupt:
        return 0


def enable() -> int:
    """Enable auto-start on boot."""
    print("Enabling auto-start on boot...")
    
    if not LAUNCHD_PLIST.exists():
        print("✗ Service not installed. Run 'install' first.")
        return 1
    
    try:
        run_command(["launchctl", "load", str(LAUNCHD_PLIST)])
        print("✓ Auto-start enabled")
        return 0
    except subprocess.CalledProcessError:
        print("✗ Failed to enable auto-start")
        return 1


def disable() -> int:
    """Disable auto-start on boot."""
    print("Disabling auto-start on boot...")
    
    if not LAUNCHD_PLIST.exists():
        print("✗ Service not installed")
        return 1
    
    try:
        run_command(["launchctl", "unload", str(LAUNCHD_PLIST)])
        print("✓ Auto-start disabled")
        return 0
    except subprocess.CalledProcessError:
        print("✗ Failed to disable auto-start")
        return 1


def config() -> int:
    """Show configuration."""
    print("\n" + "=" * 60)
    print("  Aster Service Configuration")
    print("=" * 60 + "\n")
    
    print(f"Project Root:     {PROJECT_ROOT}")
    print(f"Python Binary:    {PYTHON_BIN}")
    print(f"Config File:      {CONFIG_FILE}")
    print(f"Log Directory:    {LOG_DIR}")
    print(f"Log File:         {LOG_FILE}")
    print(f"Error Log:        {ERROR_LOG}")
    print(f"LaunchD Label:    {LAUNCHD_LABEL}")
    print(f"LaunchD Plist:    {LAUNCHD_PLIST}")
    
    print(f"\nService Configuration:")
    for key, value in SERVICE_CONFIG.items():
        print(f"  {key}: {value}")
    
    print(f"\nStatus:")
    if LAUNCHD_PLIST.exists():
        print(f"  ✓ Service installed")
    else:
        print(f"  ✗ Service not installed")
    
    print()
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aster Daemon Manager - Control background service on macOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install and enable auto-start
  python daemon.py install
  
  # Start the service
  python daemon.py start
  
  # Check status
  python daemon.py status
  
  # View logs
  python daemon.py logs
  
  # Restart service
  python daemon.py restart
  
  # Disable auto-start
  python daemon.py disable
        """,
    )
    
    parser.add_argument(
        "action",
        choices=["install", "uninstall", "start", "stop", "restart", "status", "logs", "enable", "disable", "config"],
        help="Action to perform",
    )
    
    args = parser.parse_args()
    
    actions = {
        "install": install,
        "uninstall": uninstall,
        "start": start,
        "stop": stop,
        "restart": restart,
        "status": status,
        "logs": logs,
        "enable": enable,
        "disable": disable,
        "config": config,
    }
    
    return actions[args.action]()


if __name__ == "__main__":
    sys.exit(main())
