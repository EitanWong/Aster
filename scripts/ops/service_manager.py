#!/usr/bin/env python3
"""
Aster Service Manager - Control individual services (LLM, ASR, TTS)

Usage:
    python service_manager.py list              # List all services
    python service_manager.py enable ASR        # Enable ASR service
    python service_manager.py disable TTS       # Disable TTS service
    python service_manager.py status            # Show all service status
    python service_manager.py config            # Show service configuration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"


class ServiceManager:
    """Manage Aster services configuration."""

    def __init__(self, config_path: Path = CONFIG_FILE) -> None:
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML."""
        if not self.config_path.exists():
            print(f"✗ Config file not found: {self.config_path}", file=sys.stderr)
            sys.exit(1)
        
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _save_config(self) -> None:
        """Save configuration to YAML."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def list_services(self) -> int:
        """List all available services."""
        print("\n" + "=" * 60)
        print("  Available Services")
        print("=" * 60 + "\n")
        
        services = {
            "LLM": "Large Language Model inference",
            "ASR": "Automatic Speech Recognition",
            "TTS": "Text-to-Speech synthesis",
        }
        
        for service, description in services.items():
            print(f"  {service:10} - {description}")
        
        print()
        return 0

    def get_service_status(self, service: str) -> bool:
        """Get service enabled status."""
        service_lower = service.lower()
        
        if service_lower == "llm":
            return True  # LLM is always enabled
        elif service_lower == "asr":
            return self.config.get("audio", {}).get("asr_enabled", False)
        elif service_lower == "tts":
            return self.config.get("audio", {}).get("tts_enabled", False)
        else:
            raise ValueError(f"Unknown service: {service}")

    def enable_service(self, service: str) -> int:
        """Enable a service."""
        service_lower = service.lower()
        
        if service_lower == "llm":
            print("✗ LLM service is always enabled")
            return 1
        
        if service_lower not in ["asr", "tts"]:
            print(f"✗ Unknown service: {service}")
            return 1
        
        # Ensure audio config exists
        if "audio" not in self.config:
            self.config["audio"] = {}
        
        key = f"{service_lower}_enabled"
        
        if self.config["audio"].get(key, False):
            print(f"✓ {service.upper()} is already enabled")
            return 0
        
        self.config["audio"][key] = True
        self._save_config()
        print(f"✓ {service.upper()} service enabled")
        print(f"  Restart the service for changes to take effect:")
        print(f"  python scripts/ops/aster daemon restart")
        return 0

    def disable_service(self, service: str) -> int:
        """Disable a service."""
        service_lower = service.lower()
        
        if service_lower == "llm":
            print("✗ LLM service cannot be disabled")
            return 1
        
        if service_lower not in ["asr", "tts"]:
            print(f"✗ Unknown service: {service}")
            return 1
        
        # Ensure audio config exists
        if "audio" not in self.config:
            self.config["audio"] = {}
        
        key = f"{service_lower}_enabled"
        
        if not self.config["audio"].get(key, False):
            print(f"✓ {service.upper()} is already disabled")
            return 0
        
        self.config["audio"][key] = False
        self._save_config()
        print(f"✓ {service.upper()} service disabled")
        print(f"  Restart the service for changes to take effect:")
        print(f"  python scripts/ops/aster daemon restart")
        return 0

    def show_status(self) -> int:
        """Show status of all services."""
        print("\n" + "=" * 60)
        print("  Service Status")
        print("=" * 60 + "\n")
        
        services = ["LLM", "ASR", "TTS"]
        
        for service in services:
            try:
                enabled = self.get_service_status(service)
                status = "✓ Enabled" if enabled else "✗ Disabled"
                print(f"  {service:10} {status}")
            except ValueError as e:
                print(f"  {service:10} Error: {e}")
        
        print()
        return 0

    def show_config(self) -> int:
        """Show service configuration."""
        print("\n" + "=" * 60)
        print("  Service Configuration")
        print("=" * 60 + "\n")
        
        audio_config = self.config.get("audio", {})
        
        print("Audio Services:")
        print(f"  ASR Enabled:  {audio_config.get('asr_enabled', False)}")
        print(f"  ASR Backend:  {audio_config.get('asr_backend', 'mlx')}")
        print(f"  ASR Model:    {audio_config.get('asr_model', 'N/A')}")
        print(f"  ASR Path:     {audio_config.get('asr_model_path', 'N/A')}")
        print()
        print(f"  TTS Enabled:  {audio_config.get('tts_enabled', False)}")
        print(f"  TTS Backend:  {audio_config.get('tts_backend', 'mlx')}")
        print(f"  TTS Model:    {audio_config.get('tts_model', 'N/A')}")
        print(f"  TTS Path:     {audio_config.get('tts_model_path', 'N/A')}")
        print()
        
        model_config = self.config.get("model", {})
        print("LLM Model:")
        print(f"  Name:         {model_config.get('name', 'N/A')}")
        print(f"  Path:         {model_config.get('path', 'N/A')}")
        print(f"  Runtime:      {model_config.get('runtime', 'N/A')}")
        vllm_config = self.config.get("vllm_mlx", {})
        if model_config.get("runtime") == "vllm_mlx":
            print(f"  vLLM URL:     {vllm_config.get('base_url', 'N/A')}")
        print()
        
        return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aster Service Manager - Control individual services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available services
  python service_manager.py list
  
  # Enable ASR service
  python service_manager.py enable ASR
  
  # Disable TTS service
  python service_manager.py disable TTS
  
  # Show all service status
  python service_manager.py status
  
  # Show service configuration
  python service_manager.py config
        """,
    )
    
    parser.add_argument(
        "action",
        choices=["list", "enable", "disable", "status", "config"],
        help="Action to perform",
    )
    
    parser.add_argument(
        "service",
        nargs="?",
        choices=["LLM", "ASR", "TTS"],
        help="Service name (required for enable/disable)",
    )
    
    args = parser.parse_args()
    
    manager = ServiceManager()
    
    if args.action == "list":
        return manager.list_services()
    elif args.action == "enable":
        if not args.service:
            print("✗ Service name required for enable action")
            return 1
        return manager.enable_service(args.service)
    elif args.action == "disable":
        if not args.service:
            print("✗ Service name required for disable action")
            return 1
        return manager.disable_service(args.service)
    elif args.action == "status":
        return manager.show_status()
    elif args.action == "config":
        return manager.show_config()
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
