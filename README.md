# tiny-infra

Minimal LLM inference toolkit with HuggingFace and PyTorch implementations, focused on throughput benchmarking.

## Installation

```bash
git clone git@github.com:Ruihan11/tiny-infra.git
cd tiny-infra
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -e .
```

## Authentication

Login to HuggingFace with your token:

```bash
huggingface-cli login
```

## Usage

### Throughput Benchmarking

Benchmark HuggingFace Transformers implementation:

```bash
tinyinfra benchmark throughput --model meta-llama/Meta-Llama-3-8B --wrapper hf
```

Benchmark custom PyTorch implementation:

```bash
tinyinfra benchmark throughput --model meta-llama/Meta-Llama-3-8B --wrapper customized
```

Customize benchmark parameters:

```bash
tinyinfra benchmark throughput \
    --model meta-llama/Meta-Llama-3-8B \
    --wrapper hf \
    --batch-size 8 \
    --num-tokens 256 \
    --num-runs 20 \
    --output results/my_benchmark
```

### System Information

Check CUDA availability and system info:

```bash
tinyinfra info
```

## Benchmark Results

Example results on A100 GPU:

| Wrapper | Throughput (tokens/sec) | Mean Latency (ms) |
|---------|------------------------|-------------------|
| HuggingFace | 284.08 | 7209.24 |
| PyTorch Custom | 427.19 | 4794.09 |

## Features

- **Two inference implementations:**
  - HuggingFace Transformers wrapper
  - Custom PyTorch implementation with KV cache and FlashAttention

- **Throughput benchmarking:**
  - Configurable batch size, token count, and runs
  - Warmup runs for stable measurements
  - PyTorch profiler support
  - JSON output with detailed metrics

- **CLI interface:**
  - Simple command-line interface
  - System info checking
  - Multiple device support (CUDA/CPU/auto)
