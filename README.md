# tiny-infra

Minimal LLM inference toolkit with HuggingFace and PyTorch implementations, focused on throughput benchmarking.

## Installation

```bash
git clone git@github.com:Ruihan11/tiny-infra.git
cd tiny-infra
pip install uv
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -e .
```

## Authentication

Login to HuggingFace with your token:

```bash
huggingface-cli login
```

```bash
hf download meta-llama/Meta-Llama-3-8B --local-dir meta-llama/Meta-Llama-3-8B
```

## Usage

### Throughput Benchmarking

```bash
tinyinfra benchmark throughput \
    --model meta-llama/Meta-Llama-3-8B \
    --wrapper llama3_02_opt \
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

```bash
tinyinfra benchmark throughput \
    --model meta-llama/Meta-Llama-3-8B \
    --wrapper llama3_02_opt \
    --batch-size 8 \
    --num-tokens 256 \
    --num-runs 10

tinyinfra benchmark throughput \
    --model meta-llama/Meta-Llama-3-8B \
    --wrapper llama3_01_naive \
    --batch-size 8 \
    --num-tokens 256 \
    --num-runs 10

tinyinfra benchmark throughput \
    --model meta-llama/Meta-Llama-3-8B \
    --wrapper llama3_00_hf \
    --batch-size 8 \
    --num-tokens 256 \
    --num-runs 10
```

| Wrapper | Throughput (tokens/sec) | Mean Latency (ms) |
|---------|------------------------|-------------------|
| HF | 508.19 | 4029.96 |
| Naive | 509.82 | 4017.13 |
| Opt | 641.36  | 3193.20 |

