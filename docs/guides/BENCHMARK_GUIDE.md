# Aster Benchmark Suite - Performance Testing Guide

## Overview

The Aster Benchmark Suite provides comprehensive performance testing for:
- **LLM** (Large Language Model) inference
- **ASR** (Automatic Speech Recognition)
- **TTS** (Text-to-Speech synthesis)

Measures key metrics:
- **Throughput**: tokens/sec, characters/sec, real-time factor
- **Latency**: first token time, end-to-end time
- **Resource Usage**: memory, CPU, power efficiency
- **System Info**: CPU, memory, platform details

## Quick Start

### Prerequisites

Make sure the Aster service is running:

```bash
python scripts/ops/daemon.py status
```

### Run Full Benchmark

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
python scripts/benchmark.py
```

### Save Results to File

```bash
python scripts/benchmark.py --output benchmark_results.json
```

## Benchmark Tests

### LLM Benchmarks

Tests three different prompt complexities:

1. **Short Prompt** - "What is 2+2?"
   - Max tokens: 50
   - Tests: Quick response time

2. **Medium Prompt** - "Explain quantum computing in simple terms."
   - Max tokens: 100
   - Tests: Moderate complexity

3. **Long Prompt** - Detailed ML explanation
   - Max tokens: 150
   - Tests: Complex reasoning

**Metrics:**
- `tokens_per_second` - Generation throughput
- `time_to_first_token_ms` - Latency to first token
- `total_time_ms` - End-to-end time
- `memory_peak_mb` - Peak memory usage
- `cpu_percent_avg` - Average CPU usage

### ASR Benchmarks

Tests speech recognition with real audio file:

- **Audio File**: `/Users/eitan/Desktop/PromptAudio.wav`
- **Metrics**:
  - `real_time_factor` - Audio duration / transcription time (higher is better)
  - `transcription_time_ms` - Total processing time
  - `audio_duration_sec` - Audio length
  - `memory_peak_mb` - Peak memory usage
  - `cpu_percent_avg` - Average CPU usage

**Real-Time Factor Interpretation:**
- RTF < 1.0 = Faster than real-time (excellent)
- RTF = 1.0 = Real-time performance
- RTF > 1.0 = Slower than real-time

### TTS Benchmarks

Tests text-to-speech synthesis with three text lengths:

1. **Short Text** - "Hello world"
   - Tests: Quick synthesis

2. **Medium Text** - Multi-sentence paragraph
   - Tests: Moderate complexity

3. **Long Text** - Multi-paragraph text
   - Tests: Extended synthesis

**Metrics:**
- `characters_per_second` - Synthesis throughput
- `synthesis_time_ms` - Total processing time
- `audio_duration_sec` - Generated audio length
- `memory_peak_mb` - Peak memory usage
- `cpu_percent_avg` - Average CPU usage

## Output Format

### Console Output

```
======================================================================
  Aster Comprehensive Benchmark Suite
======================================================================

System Information:
  CPU Cores:        8
  CPU Freq:         3.50 GHz
  Memory:           16.00 GB
  Available:        8.50 GB
  Platform:         darwin

Running LLM Benchmarks...
  Short prompt... ✓ 45.23 tok/s
  Medium prompt... ✓ 38.15 tok/s
  Long prompt... ✓ 32.87 tok/s

Running ASR Benchmarks...
  Testing with PromptAudio.wav... ✓ RTF: 0.85x

Running TTS Benchmarks...
  Short text... ✓ 125.34 char/s
  Medium text... ✓ 98.76 char/s
  Long text... ✓ 87.23 char/s

======================================================================
  Benchmark Summary
======================================================================

LLM Performance:
  Short prompt         45.23 tok/s  TTFT:   125.3ms  Memory:  512.5MB
  Medium prompt        38.15 tok/s  TTFT:   145.2ms  Memory:  548.3MB
  Long prompt          32.87 tok/s  TTFT:   167.8ms  Memory:  612.1MB

ASR Performance:
  ASR Test             RTF:   0.85x  Time:  1850.5ms  Memory:  256.3MB

TTS Performance:
  Short text          125.34 char/s  Time:    79.5ms  Memory:  128.7MB
  Medium text          98.76 char/s  Time:   234.2ms  Memory:  145.2MB
  Long text            87.23 char/s  Time:   456.8ms  Memory:  167.5MB
```

### JSON Output

```json
{
  "system_info": {
    "cpu_count": 8,
    "cpu_freq_ghz": 3.5,
    "memory_gb": 16.0,
    "memory_available_gb": 8.5,
    "platform": "darwin",
    "timestamp": "2026-03-19T17:45:00"
  },
  "llm_benchmarks": [
    {
      "test_name": "Short prompt",
      "prompt": "What is 2+2?",
      "prompt_tokens": 4,
      "completion_tokens": 8,
      "total_tokens": 12,
      "time_to_first_token_ms": 125.3,
      "total_time_ms": 265.2,
      "tokens_per_second": 45.23,
      "memory_peak_mb": 512.5,
      "memory_avg_mb": 480.2,
      "cpu_percent_avg": 85.5,
      "success": true
    }
  ],
  "asr_benchmarks": [...],
  "tts_benchmarks": [...]
}
```

## Performance Interpretation

### LLM Performance

**Good Performance:**
- `tokens_per_second` > 30 tok/s
- `time_to_first_token_ms` < 200ms
- `memory_peak_mb` < 1000MB

**Excellent Performance:**
- `tokens_per_second` > 50 tok/s
- `time_to_first_token_ms` < 100ms
- `memory_peak_mb` < 800MB

### ASR Performance

**Good Performance:**
- `real_time_factor` < 1.0 (faster than real-time)
- `transcription_time_ms` < 5000ms for typical audio

**Excellent Performance:**
- `real_time_factor` < 0.5 (2x faster than real-time)
- `transcription_time_ms` < 2000ms for typical audio

### TTS Performance

**Good Performance:**
- `characters_per_second` > 50 char/s
- `synthesis_time_ms` < 500ms for typical text

**Excellent Performance:**
- `characters_per_second` > 100 char/s
- `synthesis_time_ms` < 200ms for typical text

## Advanced Usage

### Custom API URL

```bash
python scripts/benchmark.py --api-url http://localhost:9000
```

### Save Results with Timestamp

```bash
python scripts/benchmark.py --output "benchmark_$(date +%Y%m%d_%H%M%S).json"
```

### Run Benchmarks Periodically

```bash
# Run every hour
while true; do
    python scripts/benchmark.py --output "benchmark_$(date +%Y%m%d_%H%M%S).json"
    sleep 3600
done
```

## Comparing Results

### Load and Compare JSON Results

```python
import json
from pathlib import Path

# Load results
with open("benchmark_results.json") as f:
    results = json.load(f)

# Extract LLM metrics
for test in results["llm_benchmarks"]:
    print(f"{test['test_name']}: {test['tokens_per_second']:.2f} tok/s")
```

### Track Performance Over Time

```bash
# Create results directory
mkdir -p benchmark_results

# Run benchmarks daily
python scripts/benchmark.py --output "benchmark_results/$(date +%Y%m%d).json"

# Compare results
ls -lh benchmark_results/
```

## Troubleshooting

### Service Not Running

```bash
# Check service status
python scripts/ops/daemon.py status

# Start service
python scripts/ops/daemon.py start
```

### Benchmark Fails

```bash
# Check API health
python scripts/ops/daemon.py health

# View service logs
python scripts/ops/daemon.py logs

# Restart service
python scripts/ops/daemon.py restart
```

### Audio File Not Found

```bash
# Check if audio file exists
ls -lh /Users/eitan/Desktop/PromptAudio.wav

# If not, create a test audio file or update the path in benchmark.py
```

## Performance Optimization Tips

### Improve LLM Performance

1. **Reduce max_tokens** - Smaller outputs are faster
2. **Use shorter prompts** - Less context to process
3. **Enable speculative decoding** - If available
4. **Increase batch size** - For multiple requests

### Improve ASR Performance

1. **Use shorter audio** - Faster processing
2. **Reduce audio quality** - Lower sample rate if acceptable
3. **Disable unused services** - Free up resources

### Improve TTS Performance

1. **Use shorter text** - Faster synthesis
2. **Reduce audio quality** - Lower sample rate if acceptable
3. **Disable unused services** - Free up resources

## System Requirements

- **CPU**: Multi-core processor (4+ cores recommended)
- **Memory**: 8GB+ RAM
- **Storage**: 20GB+ for models
- **Network**: For API communication (localhost only)

## Benchmark Results Storage

Results are saved to:
```
/Users/eitan/Documents/Projects/Python/Aster/benchmark_results/
```

Each result file contains:
- System information
- Benchmark results for each test
- Timestamps
- Success/failure status

## Next Steps

1. **Run baseline benchmark** - Establish performance baseline
2. **Monitor over time** - Track performance changes
3. **Optimize configuration** - Adjust settings based on results
4. **Compare with targets** - Set performance goals
5. **Document findings** - Keep records for reference

## Support

For issues or questions:

1. Check service status: `python scripts/ops/daemon.py status`
2. View logs: `python scripts/ops/daemon.py logs`
3. Run health check: `python scripts/ops/daemon.py health`
4. Review benchmark output for error messages
