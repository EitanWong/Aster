# Aster Background Service - Complete Setup Guide

## 🎯 Overview

Aster is now a complete macOS background service that:
- ✅ Runs silently in the background 24/7
- ✅ Auto-starts on boot
- ✅ Provides LLM, ASR, and TTS capabilities
- ✅ Optimized for performance and power efficiency
- ✅ Highly stable with auto-recovery
- ✅ Easy management via CLI tools

## 🚀 Quick Start (5 minutes)

### Step 1: Run Setup Script

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
bash scripts/ops/setup.sh
```

This opens an interactive menu. Choose option **1** to install as background service.

### Step 2: Start the Service

```bash
python scripts/ops/daemon.py start
```

### Step 3: Verify It's Running

```bash
python scripts/ops/daemon.py status
```

That's it! The service is now running and will auto-start on next boot.

## 📋 Available Commands

### Daemon Management

```bash
# Install as background service (auto-start on boot)
python scripts/ops/daemon.py install

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

# Uninstall the service
python scripts/ops/daemon.py uninstall
```

### Service Configuration

```bash
# List available services
python scripts/ops/service_manager.py list

# Enable ASR (Speech Recognition)
python scripts/ops/service_manager.py enable ASR

# Disable TTS (Text-to-Speech)
python scripts/ops/service_manager.py disable TTS

# Show service status
python scripts/ops/service_manager.py status

# Show service configuration
python scripts/ops/service_manager.py config
```

### Health Monitoring

```bash
# Run single health check
python scripts/ops/health_monitor.py check

# Run continuous monitoring (auto-recovery)
python scripts/ops/health_monitor.py monitor

# Show monitor status
python scripts/ops/health_monitor.py status
```

### Unified CLI (Recommended)

```bash
# Make the unified CLI executable
chmod +x scripts/ops/aster

# Or add to PATH for global access
export PATH="/Users/eitan/Documents/Projects/Python/Aster/scripts/ops:$PATH"

# Then use:
aster daemon install
aster daemon start
aster daemon status
aster service enable ASR
aster health
aster test
```

## 🔧 Configuration

### Enable/Disable Services

Services can be enabled or disabled without restarting:

```bash
# Enable ASR
python scripts/ops/service_manager.py enable ASR

# Disable TTS
python scripts/ops/service_manager.py disable TTS

# Restart for changes to take effect
python scripts/ops/daemon.py restart
```

### Modify Configuration

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

Then restart:

```bash
python scripts/ops/daemon.py restart
```

## 📊 Monitoring

### Check Service Health

```bash
python scripts/ops/daemon.py health
```

### View Logs

```bash
# View live logs
python scripts/ops/daemon.py logs

# Or check log files directly
tail -f logs/aster.log
tail -f logs/aster.error.log
```

### Monitor Continuously

```bash
# Run continuous monitoring with auto-recovery
python scripts/ops/health_monitor.py monitor

# Check every 60 seconds instead of default 30
python scripts/ops/health_monitor.py monitor --interval 60
```

## 🧪 Testing

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
python scripts/test_asr_interactive.py --record 5 --transcribe

# Test TTS
python scripts/test_audio_cli.py --tts "Hello world"

# Test LLM
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-9B", "messages": [{"role": "user", "content": "Hello"}]}'
```

## 🔄 Auto-Recovery

The service includes automatic recovery:

1. **Health Monitoring**: Checks service health every 30 seconds
2. **Failure Detection**: Detects when service becomes unresponsive
3. **Auto-Restart**: Automatically restarts service after 3 consecutive failures
4. **Logging**: All recovery attempts are logged

To run continuous monitoring:

```bash
python scripts/ops/health_monitor.py monitor
```

## 📁 File Locations

```
Project Root:
  /Users/eitan/Documents/Projects/Python/Aster/

Configuration:
  configs/config.yaml

Logs:
  logs/aster.log
  logs/aster.error.log
  logs/monitor.log

LaunchD Service:
  ~/Library/LaunchAgents/com.local.aster.daemon.plist

Management Scripts:
  scripts/ops/daemon.py
  scripts/ops/service_manager.py
  scripts/ops/health_monitor.py
  scripts/ops/aster
  scripts/ops/setup.sh
```

## 🎛️ Performance Tuning

### CPU Priority

The service runs with low priority (nice level 10) to avoid interfering with user tasks.

### Memory Usage

Monitor memory:

```bash
ps aux | grep aster
```

### Disable Unused Services

Disable services you don't need:

```bash
python scripts/ops/service_manager.py disable TTS
python scripts/ops/daemon.py restart
```

## 🆘 Troubleshooting

### Service Won't Start

```bash
# Check configuration
python scripts/ops/daemon.py config

# Check logs
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

# Restart
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
# Check log size
ls -lh logs/

# Clear old logs
rm logs/aster.log logs/aster.error.log

# Restart
python scripts/ops/daemon.py restart
```

## 🔐 Security

The service:
- Runs with user privileges (not root)
- Listens only on localhost (127.0.0.1)
- Logs all activities
- Auto-recovers from failures

## 📚 Additional Resources

- **Deployment Guide**: `BACKGROUND_SERVICE.md`
- **API Documentation**: `DEPLOYMENT.md`
- **Model Setup**: `docs/MODEL_SETUP.md`
- **Quick Start**: `QUICK_START_MODELS.md`

## 🎓 Common Workflows

### First-Time Setup

```bash
# 1. Install service
python scripts/ops/daemon.py install

# 2. Start service
python scripts/ops/daemon.py start

# 3. Check status
python scripts/ops/daemon.py status

# 4. Run tests
python scripts/ops/daemon.py test
```

### Daily Operations

```bash
# Check service is running
python scripts/ops/daemon.py status

# View recent logs
python scripts/ops/daemon.py logs

# Run health check
python scripts/ops/daemon.py health
```

### Maintenance

```bash
# Update configuration
# Edit configs/config.yaml

# Restart service
python scripts/ops/daemon.py restart

# Run tests
python scripts/ops/daemon.py test

# Monitor for issues
python scripts/ops/health_monitor.py monitor
```

### Troubleshooting

```bash
# Check configuration
python scripts/ops/daemon.py config

# View logs
python scripts/ops/daemon.py logs

# Run health check
python scripts/ops/daemon.py health

# Restart service
python scripts/ops/daemon.py restart
```

## ✅ Verification Checklist

After setup, verify:

- [ ] Service is installed: `python scripts/ops/daemon.py config`
- [ ] Service is running: `python scripts/ops/daemon.py status`
- [ ] API is responding: `python scripts/ops/daemon.py health`
- [ ] Tests pass: `python scripts/ops/daemon.py test`
- [ ] Logs are clean: `python scripts/ops/daemon.py logs`
- [ ] Auto-start is enabled: Check `~/Library/LaunchAgents/com.local.aster.daemon.plist`

## 🎉 You're All Set!

Your Aster inference engine is now:
- ✅ Running in the background
- ✅ Auto-starting on boot
- ✅ Monitoring its own health
- ✅ Ready to serve LLM, ASR, and TTS requests

Enjoy your local AI inference engine! 🚀
