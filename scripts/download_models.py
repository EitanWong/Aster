#!/usr/bin/env python3
"""Aster model downloader CLI.

One-click model download system for Aster inference runtime.

Usage:
    python download_models.py --all
    python download_models.py --group llm
    python download_models.py --model qwen3_5_9b
    python download_models.py --list
    python download_models.py --verify-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lib.model_manager import DownloadSummary, ModelDownloader, ModelManifest


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and verify models for Aster inference runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all required models
  python download_models.py --all

  # Download only LLM models
  python download_models.py --group llm

  # Download specific model
  python download_models.py --model qwen3_5_9b

  # List available models
  python download_models.py --list

  # Verify existing models
  python download_models.py --verify-only

  # Force re-download (skip cache)
  python download_models.py --all --force
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models (required + optional)",
    )
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="Download only required models",
    )
    parser.add_argument(
        "--group",
        choices=["asr", "llm", "tts"],
        help="Download models from specific group",
    )
    parser.add_argument(
        "--model",
        help="Download specific model by key",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing models without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download (skip existing models)",
    )
    parser.add_argument(
        "--manifest",
        default="models/manifest.yaml",
        help="Path to model manifest (default: models/manifest.yaml)",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for model paths (default: current directory)",
    )

    args = parser.parse_args()

    # Resolve paths
    manifest_path = Path(args.manifest)
    base_dir = Path(args.base_dir)

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}", file=sys.stderr)
        return 1

    try:
        manifest = ModelManifest(manifest_path)
    except Exception as e:
        print(f"Error loading manifest: {e}", file=sys.stderr)
        return 1

    # Handle --list
    if args.list:
        return _handle_list(manifest)

    # Determine which models to process
    if args.model:
        models = [manifest.get_model(args.model)]
        if not models[0]:
            print(f"Error: Model not found: {args.model}", file=sys.stderr)
            return 1
    elif args.group:
        models = manifest.get_models(group=args.group)
    elif args.required_only:
        models = manifest.get_models(required_only=True)
    elif args.all:
        models = manifest.get_models()
    else:
        # Default: required models only
        models = manifest.get_models(required_only=True)

    if not models:
        print("No models selected.", file=sys.stderr)
        return 1

    # Create downloader
    downloader = ModelDownloader(manifest, base_dir=base_dir)

    # Process models
    results = []
    if args.verify_only:
        print(f"Verifying {len(models)} model(s)...\n")
        for model in models:
            result = downloader.verify_model(model)
            results.append(result)
            _print_result(result)
    else:
        print(f"Downloading {len(models)} model(s)...\n")
        for model in models:
            result = downloader.download_model(model, force=args.force)
            results.append(result)
            _print_result(result)

    # Print summary
    summary = DownloadSummary(results)
    summary.print_summary()

    # Exit with error if any failed
    return 1 if summary.has_failures() else 0


def _handle_list(manifest: ModelManifest) -> int:
    """List available models."""
    print("\nAvailable models in manifest:\n")

    for group in ["asr", "llm", "tts"]:
        models = manifest.get_models(group=group)
        if not models:
            continue

        print(f"{'=' * 70}")
        print(f"{group.upper()} Models")
        print(f"{'=' * 70}")

        for model in models:
            required_str = "[REQUIRED]" if model.required else "[optional]"
            print(f"\n  {model.key}")
            print(f"    Name:        {model.name}")
            print(f"    Description: {model.description}")
            print(f"    Purpose:     {model.purpose}")
            print(f"    Status:      {required_str}")
            print(f"    Source:      {model.source}")
            print(f"    Repo:        {model.repo_id}")
            print(f"    Target:      {model.target_path}")
            print(f"    Size:        ~{model.size_gb}GB")
            if model.notes:
                print(f"    Notes:       {model.notes}")

        print()

    return 0


def _print_result(result) -> None:
    """Print result of a single operation."""
    if result.status == "downloaded":
        size_gb = (result.size_bytes or 0) / (1024**3)
        duration = f"{result.duration_s:.1f}s" if result.duration_s else "?"
        print(f"✓ {result.key:30s} downloaded ({size_gb:.2f}GB, {duration})")
    elif result.status == "skipped":
        size_gb = (result.size_bytes or 0) / (1024**3)
        print(f"⊘ {result.key:30s} skipped ({size_gb:.2f}GB, already exists)")
    elif result.status == "verified":
        size_gb = (result.size_bytes or 0) / (1024**3)
        if result.error:
            print(f"⚠ {result.key:30s} verified ({size_gb:.2f}GB) - {result.error}")
        else:
            print(f"✓ {result.key:30s} verified ({size_gb:.2f}GB)")
    elif result.status == "failed":
        print(f"✗ {result.key:30s} failed - {result.error}")


if __name__ == "__main__":
    sys.exit(main())
