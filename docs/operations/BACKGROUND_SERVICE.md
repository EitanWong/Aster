# Aster Background Service Setup Guide

## Overview

Aster can run as a macOS background service that automatically starts on boot and runs silently in the background. This guide covers installation, configuration, and management.

## Quick Start

### 1. Install as Background Service

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
python scripts/ops/daemon.py install
```

This will:
- Create a launchd plist file in `~/Library/LaunchAgents/`
- Register the service with macOS
- Enable auto-start on boot

### 2. Start the Service

```bash
python scripts/ops/daemon.py start
```

### 3. Check Status

```bash
python scripts/ops/daemon.py status
```

### 4. View Logs

```bash
python scripts/ops/daemon.py logs
```

## Service Management

### Daemon Commands

```bash
# Install as background service
python scripts/ops/daemon.py install

# Uninstall the service
python scripts/ops/daemon.py uninstall

# Start the service
python scripts/ops/daemon.py start

# Stop the service
python scripts/ops/daemon.py stop

# Restart the service
python scripts/ops/daemon.py restart

# Check service status
python scripts/ops/daemon.py status

# View live logs
python scripts/ops/daemon.py logs

# Enable auto-start on boot
python scripts/ops/daemon.py enable

# Disable auto-start on boot
python scripts/ops/daemon.py disable

# Show configuration
python scripts/ops/daemon.py config
```

## Service Configuration

### Enable/Disable Individual Services

```bash
# List available services
python scripts/ops/service_manager.py list

# Enable ASR service
python scripts/ops/service_manager.py enable ASR

# Disable TTS service
python scripts/ops/service_manager.py disable TTS

# Show service status
python scripts/ops/service_manager.py status

# Show service configuration
python scripts/ops/service_manager.py config
```

### Available Services

- **LLM** - Large Language Model inference (always enabled)
- **ASR** - Automatic Speech Recognition (optional)
- **TTS** - Text-to-Speech synthesis (optional)

## Unified CLI

For convenience, use the unified `aster` command:

```bash
# Make it executable
chmod +x scripts/ops/aster

# Or add to PATH
export PATH="/Users/eitan/Documents/Projects/Python/Aster/scripts/ops:$PATH"

# Then use:
aster daemon install
aster daemon start
aster daemon status
aster service enable ASR
aster health
aster test
```

## Configuration Files

### Main Configuration

- **Location**: `configs/config.yaml`
- **Controls**: Model paths, API settings, audio services

### Service Logs

- **Standard Output**: `logs/aster.log`
- **Error Output**: `logs/aster.error.log`
- **View**: `python scripts/ops/daemon.py logs`

### LaunchD Configuration

- **Location**: `~/Library/LaunchAgents/com.local.aster.daemon.plist`
- **Auto-start**: Enabled by default
- **Restart Policy**: Automatic restart on failure

## Performance & Power Management

### CPU Usage

The service is configured with:
- **Nice Level**: 10 (low priority, won't interfere with user tasks)
- **Process Type**: Background (optimized for background execution)

### Memory Usage

Monitor memory usage:

```bash
# Check process memory
ps aux | grep aster

# Or use Activity Monitor
open -a "Activity Monitor"
```

### Power Management

The service respects macOS power management:
- Continues running in background
- Respects system sleep settings
- Can be paused/resumed as needed

## Troubleshooting

### Service Won't Start

```bash
# Check if service is installed
python scripts/ops/daemon.py config

# Check logs for errors
python scripts/ops/daemon.py logs

# Try restarting
python scripts/ops/daemon.py restart
```

### API Not Responding

```bash
# Check health
python scripts/ops/daemon.py health

# Check if service is running
python scripts/ops/daemon.py status

# Restart service
python scripts/ops/daemon.py restart
```

### High CPU Usage

```bash
# Check which service is consuming CPU
ps aux | grep aster

# Disable unused services
python scripts/ops/service_manager.py disable TTS

# Restart
python scripts/ops/daemon.py restart
```

### Logs Growing Too Large

```bash
# View current log size
ls -lh logs/

# Clear old logs
rm logs/aster.log logs/aster.error.log

# Restart service
python scripts/ops/daemon.py restart
```

## Advanced Configuration

### Custom Config File

Edit `configs/config.yaml` to customize:

```yaml
api:
  host: 127.0.0.1
  port: 8080

model:
  name: Qwen3.5-9B
  path: /path/to/model

audio:
  asr_enabled: true
  tts_enabled: true
```

### Disable Auto-Start

```bash
python scripts/ops/daemon.py disable
```

### Manual Start/Stop

```bash
# Start manually
python scripts/ops/daemon.py start

# Stop manually
python scripts/ops/daemon.py stop
```

## Testing

### Quick Health Check

```bash
python scripts/ops/daemon.py health
```

### Run Full Test Suite

```bash
python scripts/ops/daemon.py test
```

### Test Individual Services

```bash
# Test ASR
python scripts/test_asr_interactive.py --file audio.wav

# Test TTS
python scripts/test_audio_cli.py --tts "Hello"

# Test LLM
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-9B", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Uninstall

To completely remove the service:

```bash
# Stop the service
python scripts/ops/daemon.py stop

# Uninstall
python scripts/ops/daemon.py uninstall

# Verify removal
python scripts/ops/daemon.py status
```

## Best Practices

1. **Regular Monitoring**: Check logs periodically for errors
2. **Resource Management**: Monitor CPU and memory usage
3. **Service Updates**: Restart after updating configuration
4. **Backup Logs**: Archive logs before clearing
5. **Test After Changes**: Run health check after configuration changes

## Support

For issues or questions:

1. Check logs: `python scripts/ops/daemon.py logs`
2. Check status: `python scripts/ops/daemon.py status`
3. Review configuration: `python scripts/ops/daemon.py config`
4. Run tests: `python scripts/ops/daemon.py test`
