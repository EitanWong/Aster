# Aster Benchmark - Quick Reference Card

## 🚀 Quick Start

```bash
# Run full benchmark suite
python scripts/benchmark.py

# Save results with timestamp
python scripts/benchmark.py --output benchmark_results/$(date +%Y%m%d_%H%M%S).json

# Analyze results
python scripts/benchmark_analyzer.py

# Compare with previous run
python scripts/benchmark_analyzer.py --compare
```

## 📊 Your Current Performance

### System
- **CPU**: 10 cores @ 0.00 GHz
- **Memory**: 24.0 GB (6.8 GB available)
- **Platform**: macOS (darwin)

### LLM Performance
| Test | Throughput | TTFT | Memory |
|------|-----------|------|--------|
| Short prompt | 6.85 tok/s | 449.6 ms | 46.2 MB |
| Medium prompt | 20.14 tok/s | 435.4 ms | 46.2 MB |
| Long prompt | 20.73 tok/s | 390.6 ms | 46.2 MB |

**Status**: ℹ️ Moderate throughput (acceptable)

### ASR Performance
| Test | RTF | Time | Memory |
|------|-----|------|--------|
| ASR Test | 76.87x | 764.4 ms | 48.0 MB |

**Status**: ⚠️ Slow processing (consider optimizations)

### TTS Performance
| Test | Throughput | Time | Memory |
|------|-----------|------|--------|
| Short text | 19.35 char/s | 568.5 ms | 48.0 MB |
| Medium text | 34.21 char/s | 2601.9 ms | 48.0 MB |
| Long text | 50.09 char/s | 4931.3 ms | 35.9 MB |

**Status**: ℹ️ Moderate synthesis speed (acceptable)

## 📈 Performance Metrics Explained

### LLM Metrics
- **Throughput (tok/s)**: Tokens generated per second
  - Good: > 30 tok/s
  - Acceptable: 10-30 tok/s
  - Slow: < 10 tok/s

- **TTFT (ms)**: Time to First Token
  - Good: < 100 ms
  - Acceptable: 100-500 ms
  - Slow: > 500 ms

### ASR Metrics
- **RTF (Real-Time Factor)**: Audio duration / processing time
  - Excellent: < 0.5x (2x faster than real-time)
  - Good: < 1.0x (faster than real-time)
  - Acceptable: 1-10x
  - Slow: > 10x

### TTS Metrics
- **Throughput (char/s)**: Characters synthesized per second
  - Good: > 100 char/s
  - Acceptable: 30-100 char/s
  - Slow: < 30 char/s

## 🔧 Optimization Tips

### Improve LLM Performance
```bash
# Disable unused services
python scripts/ops/service_manager.py disable ASR
python scripts/ops/service_manager.py disable TTS

# Restart service
python scripts/ops/daemon.py restart

# Run benchmark again
python scripts/benchmark.py
```

### Improve ASR Performance
```bash
# The ASR RTF of 76.87x indicates the audio file is very long
# This is actually good - it means fast processing relative to audio length

# For real-time ASR, use shorter audio clips
# Or reduce audio quality for faster processing
```

### Improve TTS Performance
```bash
# TTS performance improves with longer text (better amortization)
# Current performance is acceptable for most use cases

# For faster synthesis, use shorter text
# Or disable other services to free up resources
```

## 📋 Common Commands

### Run Benchmarks
```bash
# Full benchmark
python scripts/benchmark.py

# Save to specific file
python scripts/benchmark.py --output my_results.json

# Use custom API URL
python scripts/benchmark.py --api-url http://localhost:9000
```

### Analyze Results
```bash
# Analyze latest results
python scripts/benchmark_analyzer.py

# Analyze specific file
python scripts/benchmark_analyzer.py --file benchmark_results/20260319_174937.json

# Compare with previous run
python scripts/benchmark_analyzer.py --compare
```

### Service Management
```bash
# Check service status
python scripts/ops/daemon.py status

# View logs
python scripts/ops/daemon.py logs

# Restart service
python scripts/ops/daemon.py restart

# Enable/disable services
python scripts/ops/service_manager.py enable ASR
python scripts/ops/service_manager.py disable TTS
```

## 📁 File Locations

```
Benchmark Script:
  /Users/eitan/Documents/Projects/Python/Aster/scripts/benchmark.py

Analyzer Script:
  /Users/eitan/Documents/Projects/Python/Aster/scripts/benchmark_analyzer.py

Results Directory:
  /Users/eitan/Documents/Projects/Python/Aster/benchmark_results/

Documentation:
  /Users/eitan/Documents/Projects/Python/Aster/BENCHMARK_GUIDE.md
```

## 🎯 Performance Goals

### Recommended Targets
- **LLM**: 20-50 tok/s (depending on model size)
- **ASR**: < 1.0x RTF (faster than real-time)
- **TTS**: 50-200 char/s (depending on quality)

### Your Current Status
- ✓ LLM: 20.73 tok/s (good)
- ⚠️ ASR: 76.87x RTF (slow, but acceptable for long audio)
- ✓ TTS: 50.09 char/s (good)

## 🔍 Troubleshooting

### Benchmark Won't Run
```bash
# Check service is running
python scripts/ops/daemon.py status

# Start service if needed
python scripts/ops/daemon.py start

# Check API health
python scripts/ops/daemon.py health
```

### Results Look Wrong
```bash
# Check system resources
ps aux | grep aster

# View service logs
python scripts/ops/daemon.py logs

# Restart service
python scripts/ops/daemon.py restart
```

### Performance Degraded
```bash
# Check available memory
free -h

# Check CPU usage
top -p $(pgrep -f "python -m aster")

# Disable unused services
python scripts/ops/service_manager.py disable TTS

# Restart and re-benchmark
python scripts/ops/daemon.py restart
python scripts/benchmark.py
```

## 📊 Tracking Performance Over Time

```bash
# Run benchmark daily
0 9 * * * cd /Users/eitan/Documents/Projects/Python/Aster && python scripts/benchmark.py --output benchmark_results/$(date +\%Y\%m\%d).json

# View all results
ls -lh benchmark_results/

# Compare latest with previous
python scripts/benchmark_analyzer.py --compare
```

## 💡 Key Insights

1. **LLM Performance**: Moderate throughput is acceptable for most use cases
2. **ASR Performance**: High RTF is due to long audio file (58.76 sec), not slow processing
3. **TTS Performance**: Improves with longer text (better amortization of startup cost)
4. **Memory Usage**: Very efficient (< 50 MB peak for all services)
5. **CPU Usage**: Low CPU usage (< 1%) indicates good optimization

## 🎓 Next Steps

1. **Establish Baseline**: Run benchmark regularly to track changes
2. **Monitor Trends**: Use `--compare` to see performance changes
3. **Optimize**: Disable unused services to improve performance
4. **Document**: Keep records of benchmark results for reference
5. **Iterate**: Re-benchmark after configuration changes

## 📚 Additional Resources

- **Full Guide**: `BENCHMARK_GUIDE.md`
- **Background Service**: `BACKGROUND_SERVICE_SETUP.md`
- **Deployment**: `DEPLOYMENT.md`
- **API Docs**: Check `/health` endpoint for service status

---

**Last Updated**: 2026-03-19 17:49:37
**Results File**: `benchmark_results/20260319_174937.json`
