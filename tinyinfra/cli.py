"""
CLI for tinyinfra
"""
import click
import sys
import json
from pathlib import Path
import torch

from .model.hf.llama3_hf import Llama3HF
from .model.customized.llama3_customized import Llama3Customized
from .benchmark.throughput import ThroughputBenchmark


@click.group()
def cli():
    """TinyInfra: Minimal LLM Inference Toolkit"""
    pass


# ============================================================
# BENCHMARK COMMANDS
# ============================================================

@cli.group()
def benchmark():
    """Run benchmarks"""
    pass


@benchmark.command()
@click.option('--model', default='meta-llama/Meta-Llama-3-8B', help='Model name or path')
@click.option('--wrapper', default='hf', type=click.Choice(['hf', 'customized']), help='Model wrapper to use')
@click.option('--batch-size', default=1, type=int, help='Batch size')
@click.option('--num-tokens', default=128, type=int, help='Number of tokens to generate')
@click.option('--num-runs', default=10, type=int, help='Number of benchmark runs')
@click.option('--warmup', default=3, type=int, help='Number of warmup runs')
@click.option('--prompt', default='Explain artificial intelligence', help='Input prompt for generation')
@click.option('--output', default='results/baseline', help='Output directory')
@click.option('--profile', is_flag=True, help='Enable profiling (lightweight)')
@click.option('--device', default='cuda', type=click.Choice(['cuda', 'cpu', 'auto']), help='Device to use')
def throughput(model, wrapper, batch_size, num_tokens, num_runs, warmup, prompt, output, profile, device):
    """
    Measure throughput (tokens/second)

    \b
    Examples:
        # Basic usage (default HF wrapper)
        tinyinfra benchmark throughput --model meta-llama/Meta-Llama-3-8B

        # Use customized PyTorch wrapper
        tinyinfra benchmark throughput --model meta-llama/Meta-Llama-3-8B --wrapper customized

        # Batch processing
        tinyinfra benchmark throughput \\
            --model meta-llama/Meta-Llama-3-8B \\
            --batch-size 8 \\
            --num-tokens 256 \\
            --num-runs 50

        # Custom prompt
        tinyinfra benchmark throughput \\
            --model meta-llama/Meta-Llama-3-8B \\
            --prompt "Write a Python function to" \\
            --num-tokens 512

        # With profiling
        tinyinfra benchmark throughput \\
            --model meta-llama/Meta-Llama-3-8B \\
            --profile \\
            --output results/profile

        # CPU inference
        tinyinfra benchmark throughput \\
            --model meta-llama/Meta-Llama-3-8B \\
            --device cpu

    \b
    Wrappers:
        hf          - HuggingFace Transformers wrapper
        customized  - Custom PyTorch implementation
    """
    click.echo(f"\n{'='*60}")
    click.echo("THROUGHPUT BENCHMARK")
    click.echo(f"{'='*60}")

    click.echo(f"\nConfiguration:")
    click.echo(f"   Model:      {model}")
    click.echo(f"   Wrapper:    {wrapper}")
    click.echo(f"   Device:     {device}")
    click.echo(f"   Batch:      {batch_size}")
    click.echo(f"   Tokens:     {num_tokens}")
    click.echo(f"   Runs:       {num_runs}")
    click.echo(f"   Warmup:     {warmup}")
    click.echo(f"   Prompt:     {prompt[:50]}...")
    click.echo(f"   Profiling:  {'YES' if profile else 'NO'}")

    # Load model
    click.echo("\nLoading model...")
    try:
        # Handle device setting
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Select wrapper
        if wrapper == 'hf':
            model_instance = Llama3HF(model_name=model, device=device)
        elif wrapper == 'customized':
            model_instance = Llama3Customized(model_name=model, device=device)
        else:
            raise ValueError(f"Unknown wrapper: {wrapper}")

        click.echo(f"Loaded: {model_instance.get_model_size():.1f}GB")
        if device == 'cuda':
            click.echo(f"   Memory: {model_instance.get_memory_usage():.1f}GB")
    except Exception as e:
        click.echo(f"Failed: {e}", err=True)
        sys.exit(1)

    # Benchmark
    try:
        bench = ThroughputBenchmark(model_instance)
        results = bench.run(
            prompt=prompt,
            batch_size=batch_size,
            num_tokens=num_tokens,
            num_runs=num_runs,
            warmup_runs=warmup,
            enable_profiler=profile,
            profile_output_dir=output if profile else None
        )

        bench.print_results(results)

        # Save
        Path(output).mkdir(parents=True, exist_ok=True)
        with open(Path(output) / "throughput.json", 'w') as f:
            # Add config to results
            results['config'] = {
                'model': model,
                'wrapper': wrapper,
                'device': device,
                'batch_size': batch_size,
                'num_tokens': num_tokens,
                'num_runs': num_runs,
                'prompt': prompt
            }
            json.dump(results, f, indent=2)

        click.echo(f"Saved: {output}/throughput.json")

    except Exception as e:
        click.echo(f"Failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================
# INFO COMMAND
# ============================================================

@cli.command()
def info():
    """Show system info"""
    click.echo(f"\n{'='*60}")
    click.echo("SYSTEM INFO")
    click.echo(f"{'='*60}")
    click.echo(f"\nPython: {sys.version.split()[0]}")
    click.echo(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        click.echo(f"\nCUDA: {torch.cuda.get_device_name(0)}")
        click.echo(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        click.echo(f"   Compute: {torch.cuda.get_device_capability(0)}")
    else:
        click.echo(f"\nCUDA: Not available")

    click.echo()


if __name__ == '__main__':
    cli()
