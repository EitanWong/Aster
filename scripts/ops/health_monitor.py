#!/usr/bin/env python3
"""
Aster Health Monitor - Continuous monitoring and auto-recovery

Monitors the Aster service and automatically recovers from failures.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
MONITOR_LOG = LOG_DIR / "monitor.log"

# Configuration
CHECK_INTERVAL = 30  # seconds
HEALTH_TIMEOUT = 5  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def setup_logging() -> logging.Logger:
    """Setup logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("aster-monitor")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(MONITOR_LOG)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class HealthMonitor:
    """Monitor Aster service health."""

    def __init__(self, api_url: str = "http://127.0.0.1:8080", logger: logging.Logger | None = None) -> None:
        self.api_url = api_url
        self.logger = logger or setup_logging()
        self.consecutive_failures = 0
        self.last_check_time = None
        self.last_check_status = None

    def check_health(self) -> dict[str, Any]:
        """Check API health."""
        try:
            response = httpx.get(
                f"{self.api_url}/health",
                timeout=HEALTH_TIMEOUT,
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "healthy": True,
                    "status": data.get("status", "unknown"),
                    "degraded": data.get("degraded", False),
                    "details": data.get("details", {}),
                }
            else:
                return {
                    "healthy": False,
                    "status": f"HTTP {response.status_code}",
                    "degraded": True,
                    "error": f"Unexpected status code: {response.status_code}",
                }
        except httpx.ConnectError:
            return {
                "healthy": False,
                "status": "connection_error",
                "degraded": True,
                "error": "Failed to connect to API",
            }
        except httpx.TimeoutException:
            return {
                "healthy": False,
                "status": "timeout",
                "degraded": True,
                "error": "Health check timed out",
            }
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "degraded": True,
                "error": str(e),
            }

    def handle_failure(self) -> None:
        """Handle health check failure."""
        self.consecutive_failures += 1
        
        if self.consecutive_failures == 1:
            self.logger.warning(f"Health check failed (attempt {self.consecutive_failures})")
        elif self.consecutive_failures == MAX_RETRIES:
            self.logger.error(f"Health check failed {MAX_RETRIES} times, attempting recovery...")
            self.attempt_recovery()
        else:
            self.logger.warning(f"Health check failed (attempt {self.consecutive_failures}/{MAX_RETRIES})")

    def handle_success(self) -> None:
        """Handle successful health check."""
        if self.consecutive_failures > 0:
            self.logger.info(f"Service recovered after {self.consecutive_failures} failures")
        
        self.consecutive_failures = 0

    def attempt_recovery(self) -> None:
        """Attempt to recover the service."""
        import subprocess
        
        self.logger.info("Attempting service recovery...")
        
        try:
            # Try to restart the service
            result = subprocess.run(
                ["launchctl", "restart", "com.local.aster.daemon"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode == 0:
                self.logger.info("Service restart initiated")
                time.sleep(5)  # Wait for service to start
            else:
                self.logger.error(f"Failed to restart service: {result.stderr}")
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")

    def run_continuous_monitoring(self, interval: int = CHECK_INTERVAL) -> None:
        """Run continuous monitoring."""
        self.logger.info(f"Starting health monitor (check interval: {interval}s)")
        
        try:
            while True:
                self.last_check_time = datetime.now()
                health = self.check_health()
                self.last_check_status = health
                
                if health["healthy"]:
                    self.handle_success()
                    self.logger.debug(f"Health check passed: {health['status']}")
                else:
                    self.handle_failure()
                    self.logger.warning(f"Health check failed: {health.get('error', 'unknown')}")
                
                time.sleep(interval)
        except KeyboardInterrupt:
            self.logger.info("Monitor stopped by user")
        except Exception as e:
            self.logger.error(f"Monitor error: {e}")

    def run_single_check(self) -> int:
        """Run a single health check."""
        health = self.check_health()
        
        if health["healthy"]:
            print(f"✓ Service is healthy: {health['status']}")
            return 0
        else:
            print(f"✗ Service is unhealthy: {health.get('error', 'unknown')}")
            return 1

    def show_status(self) -> int:
        """Show current status."""
        print("\n" + "=" * 60)
        print("  Aster Health Monitor Status")
        print("=" * 60 + "\n")
        
        if self.last_check_time:
            print(f"Last Check:           {self.last_check_time}")
        else:
            print(f"Last Check:           Never")
        
        if self.last_check_status:
            status = self.last_check_status
            print(f"Status:               {status['status']}")
            print(f"Healthy:              {status['healthy']}")
            print(f"Degraded:             {status['degraded']}")
            
            if status.get("error"):
                print(f"Error:                {status['error']}")
        
        print(f"Consecutive Failures: {self.consecutive_failures}")
        print()
        
        return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aster Health Monitor - Continuous monitoring and auto-recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run continuous monitoring
  python health_monitor.py monitor
  
  # Run single health check
  python health_monitor.py check
  
  # Show current status
  python health_monitor.py status
  
  # Run with custom check interval
  python health_monitor.py monitor --interval 60
        """,
    )
    
    parser.add_argument(
        "action",
        choices=["monitor", "check", "status"],
        help="Action to perform",
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=CHECK_INTERVAL,
        help=f"Check interval in seconds (default: {CHECK_INTERVAL})",
    )
    
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8080",
        help="API URL (default: http://127.0.0.1:8080)",
    )
    
    args = parser.parse_args()
    
    logger = setup_logging()
    monitor = HealthMonitor(api_url=args.api_url, logger=logger)
    
    if args.action == "monitor":
        monitor.run_continuous_monitoring(interval=args.interval)
        return 0
    elif args.action == "check":
        return monitor.run_single_check()
    elif args.action == "status":
        return monitor.show_status()
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
