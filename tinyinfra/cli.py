"""
CLI for tinyinfra
"""
import click
import sys
import json
from pathlib import Path
import torch

from .model.llama3 import Llama3Model
from .benchmark.throughput import ThroughputBenchmark
from .benchmark.latency import LatencyBenchmark


@click.group()
def cli():
    """🚀 TinyInfra: Minimal LLM Inference Toolkit"""
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
@click.option('--batch-size', default=1, type=int, help='Batch size')
@click.option('--num-tokens', default=128, type=int, help='Tokens to generate')
@click.option('--num-runs', default=10, type=int, help='Benchmark runs')
@click.option('--output', default='results/baseline', help='Output directory')
@click.option('--profile', is_flag=True, help='Enable profiling (lightweight)')
def throughput(model, batch_size, num_tokens, num_runs, output, profile):
    """
    Measure throughput (tokens/second)
    
    Example:
        tinyinfra benchmark throughput
        tinyinfra benchmark throughput --profile
        tinyinfra benchmark throughput --batch-size 8 --num-runs 50
    """
    click.echo(f"\n{'='*60}")
    click.echo("🚀 THROUGHPUT BENCHMARK")
    click.echo(f"{'='*60}")
    
    # Load model
    click.echo("\n⏳ Loading model...")
    try:
        llama = Llama3Model(model_name=model)
        click.echo(f"✅ Loaded: {llama.get_model_size():.1f}GB, {llama.get_memory_usage():.1f}GB mem")
    except Exception as e:
        click.echo(f"❌ Failed: {e}", err=True)
        sys.exit(1)
    
    # Benchmark
    try:
        bench = ThroughputBenchmark(llama)
        results = bench.run(
            batch_size=batch_size,
            num_tokens=num_tokens,
            num_runs=num_runs,
            enable_profiler=profile,
            profile_output_dir=output if profile else None
        )
        
        bench.print_results(results)
        
        # Save
        Path(output).mkdir(parents=True, exist_ok=True)
        with open(Path(output) / "throughput.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        click.echo(f"💾 Saved: {output}/throughput.json")
        
    except Exception as e:
        click.echo(f"❌ Failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@benchmark.command()
@click.option('--model', default='meta-llama/Meta-Llama-3-8B', help='Model name or path')
@click.option('--num-tokens', default=128, type=int, help='Tokens to generate')
@click.option('--num-runs', default=100, type=int, help='Benchmark runs')
@click.option('--output', default='results/baseline', help='Output directory')
@click.option('--profile', is_flag=True, help='Enable profiling (lightweight)')
def latency(model, num_tokens, num_runs, output, profile):
    """
    Measure latency (response time)
    
    Example:
        tinyinfra benchmark latency
        tinyinfra benchmark latency --profile
        tinyinfra benchmark latency --num-runs 200
    """
    click.echo(f"\n{'='*60}")
    click.echo("⚡ LATENCY BENCHMARK")
    click.echo(f"{'='*60}")
    
    # Load model
    click.echo("\n⏳ Loading model...")
    try:
        llama = Llama3Model(model_name=model)
        click.echo(f"✅ Loaded: {llama.get_model_size():.1f}GB, {llama.get_memory_usage():.1f}GB mem")
    except Exception as e:
        click.echo(f"❌ Failed: {e}", err=True)
        sys.exit(1)
    
    # Benchmark
    try:
        bench = LatencyBenchmark(llama)
        results = bench.run(
            num_tokens=num_tokens,
            num_runs=num_runs,
            enable_profiler=profile,
            profile_output_dir=output if profile else None
        )
        
        bench.print_results(results)
        
        # Save
        Path(output).mkdir(parents=True, exist_ok=True)
        with open(Path(output) / "latency.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        click.echo(f"💾 Saved: {output}/latency.json")
        
    except Exception as e:
        click.echo(f"❌ Failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@benchmark.command()
@click.option('--model', default='meta-llama/Meta-Llama-3-8B', help='Model name or path')
@click.option('--output', default='results/baseline', help='Output directory')
def all(model, output):
    """
    Run all benchmarks
    
    Example:
        tinyinfra benchmark all
        tinyinfra benchmark all --model models/quantized/llama3-gptq-int8
    """
    click.echo(f"\n{'='*60}")
    click.echo("🎯 FULL BENCHMARK")
    click.echo(f"{'='*60}")
    
    try:
        llama = Llama3Model(model_name=model)
        
        # Throughput
        click.echo("\n[1/2] Throughput...")
        bench_tp = ThroughputBenchmark(llama)
        res_tp = bench_tp.run(num_runs=20)
        bench_tp.print_results(res_tp)
        
        # Latency
        click.echo("\n[2/2] Latency...")
        bench_lat = LatencyBenchmark(llama)
        res_lat = bench_lat.run(num_runs=100)
        bench_lat.print_results(res_lat)
        
        # Save
        Path(output).mkdir(parents=True, exist_ok=True)
        with open(Path(output) / "throughput.json", 'w') as f:
            json.dump(res_tp, f, indent=2)
        with open(Path(output) / "latency.json", 'w') as f:
            json.dump(res_lat, f, indent=2)
        
        click.echo(f"\n✅ Complete! Results: {output}/")
        
    except Exception as e:
        click.echo(f"❌ Failed: {e}", err=True)
        sys.exit(1)


@benchmark.command()
@click.option('--baseline', required=True, help='Baseline model path')
@click.option('--optimized', required=True, help='Optimized model path')
@click.option('--output', default='results/comparison', help='Output directory')
@click.option('--num-runs', default=20, type=int, help='Number of runs per benchmark')
def compare(baseline, optimized, output, num_runs):
    """
    Compare two models (baseline vs optimized)
    
    Example:
        tinyinfra benchmark compare \\
            --baseline meta-llama/Meta-Llama-3-8B \\
            --optimized models/quantized/Meta-Llama-3-8B-gptq-int8
    """
    click.echo(f"\n{'='*60}")
    click.echo("📊 MODEL COMPARISON")
    click.echo(f"{'='*60}")
    click.echo(f"Baseline:  {baseline}")
    click.echo(f"Optimized: {optimized}")
    click.echo(f"Runs:      {num_runs}")
    
    results = {}
    
    try:
        # Test baseline
        click.echo(f"\n[1/2] Testing baseline...")
        model_base = Llama3Model(model_name=baseline)
        
        bench_tp = ThroughputBenchmark(model_base)
        results['baseline_throughput'] = bench_tp.run(num_runs=num_runs)
        
        bench_lat = LatencyBenchmark(model_base)
        results['baseline_latency'] = bench_lat.run(num_runs=num_runs * 5)
        
        results['baseline_size_gb'] = model_base.get_model_size()
        results['baseline_memory_gb'] = model_base.get_memory_usage()
        
        # Clean up
        del model_base
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Test optimized
        click.echo(f"\n[2/2] Testing optimized...")
        model_opt = Llama3Model(model_name=optimized)
        
        bench_tp_opt = ThroughputBenchmark(model_opt)
        results['optimized_throughput'] = bench_tp_opt.run(num_runs=num_runs)
        
        bench_lat_opt = LatencyBenchmark(model_opt)
        results['optimized_latency'] = bench_lat_opt.run(num_runs=num_runs * 5)
        
        results['optimized_size_gb'] = model_opt.get_model_size()
        results['optimized_memory_gb'] = model_opt.get_memory_usage()
        
        # Print comparison
        _print_comparison(results)
        
        # Save
        Path(output).mkdir(parents=True, exist_ok=True)
        with open(Path(output) / "comparison.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        click.echo(f"\n💾 Saved: {output}/comparison.json")
        
    except Exception as e:
        click.echo(f"❌ Failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _print_comparison(results):
    """Pretty print comparison"""
    print(f"\n{'='*75}")
    print(f"📊 COMPARISON RESULTS")
    print(f"{'='*75}")
    
    base_tp = results['baseline_throughput']['throughput_tokens_per_sec']
    opt_tp = results['optimized_throughput']['throughput_tokens_per_sec']
    
    base_lat = results['baseline_latency']['end_to_end']['mean_ms']
    opt_lat = results['optimized_latency']['end_to_end']['mean_ms']
    
    base_size = results['baseline_size_gb']
    opt_size = results['optimized_size_gb']
    
    base_mem = results['baseline_memory_gb']
    opt_mem = results['optimized_memory_gb']
    
    print(f"\n{'Metric':<30} {'Baseline':<15} {'Optimized':<15} {'Change':<15}")
    print(f"{'-'*75}")
    print(f"{'Throughput (tok/s)':<30} {base_tp:<15.2f} {opt_tp:<15.2f} {_percent_change(opt_tp, base_tp):<15}")
    print(f"{'Latency (ms)':<30} {base_lat:<15.2f} {opt_lat:<15.2f} {_percent_change(opt_lat, base_lat, inverse=True):<15}")
    print(f"{'Model Size (GB)':<30} {base_size:<15.2f} {opt_size:<15.2f} {_percent_change(opt_size, base_size, inverse=True):<15}")
    print(f"{'Memory Usage (GB)':<30} {base_mem:<15.2f} {opt_mem:<15.2f} {_percent_change(opt_mem, base_mem, inverse=True):<15}")
    
    print(f"\n{'='*75}")
    
    # Summary
    avg_speedup = ((opt_tp / base_tp) + (base_lat / opt_lat)) / 2
    size_reduction = (1 - opt_size / base_size) * 100
    
    print(f"\n📈 Summary:")
    print(f"   Average Speedup:  {avg_speedup:.2f}x")
    print(f"   Size Reduction:   {size_reduction:.1f}%")
    print(f"   Memory Savings:   {(1 - opt_mem / base_mem) * 100:.1f}%")
    print(f"\n{'='*75}\n")


def _percent_change(new, old, inverse=False):
    """Calculate percentage change with arrow"""
    if old == 0:
        return "N/A"
    
    change = ((new - old) / old) * 100
    
    # Determine if improvement
    if inverse:
        is_better = change < 0
    else:
        is_better = change > 0
    
    symbol = "✓" if is_better else "✗"
    arrow = "↑" if change > 0 else "↓"
    
    return f"{symbol} {arrow} {abs(change):.1f}%"

# ============================================================
# QUANTIZATION COMMANDS
# ============================================================

@cli.group()
def quantize():
    """Quantize models (AWQ and BitsAndBytes)"""
    pass


@quantize.command()
@click.option('--model', required=True, help='Model to quantize')
@click.option('--bits', default=4, type=click.Choice(['4', '8']), help='Quantization bits')
@click.option('--output', default='models/quantized', help='Output directory')
@click.option('--group-size', default=128, type=int, help='Group size')
def awq(model, bits, output, group_size):
    """
    Quantize with AWQ (recommended for 4-bit)
    
    Example:
        tinyinfra quantize awq --model meta-llama/Meta-Llama-3-8B --bits 4
        tinyinfra quantize awq --model meta-llama/Meta-Llama-3-8B --bits 8
    """
    from .quantization.quantizer import AWQQuantizer
    
    try:
        quantizer = AWQQuantizer(model)
        output_path = quantizer.quantize(
            bits=int(bits),
            output_dir=output,
            group_size=group_size
        )
        
        click.echo(f"\n✅ Done! Next steps:")
        click.echo(f"   # Test performance")
        click.echo(f"   tinyinfra benchmark throughput --model {output_path}")
        click.echo(f"")
        click.echo(f"   # Compare with baseline")
        click.echo(f"   tinyinfra benchmark compare \\")
        click.echo(f"       --baseline {model} \\")
        click.echo(f"       --optimized {output_path}")
        
    except Exception as e:
        click.echo(f"\n❌ Quantization failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@quantize.command()
@click.option('--model', required=True, help='Model to quantize')
@click.option('--bits', default=8, type=click.Choice(['4', '8']), help='Quantization bits')
@click.option('--output', default='models/quantized', help='Output directory')
def bnb(model, bits, output):
    """
    Quantize with BitsAndBytes (simple, fast)
    
    Example:
        tinyinfra quantize bnb --model meta-llama/Meta-Llama-3-8B --bits 8
        tinyinfra quantize bnb --model meta-llama/Meta-Llama-3-8B --bits 4
    """
    from .quantization.quantizer import BitsAndBytesQuantizer
    
    try:
        quantizer = BitsAndBytesQuantizer(model)
        output_path = quantizer.quantize(
            bits=int(bits),
            output_dir=output
        )
        
        click.echo(f"\n✅ Done! Next steps:")
        click.echo(f"   # Test performance")
        click.echo(f"   tinyinfra benchmark throughput --model {output_path}")
        click.echo(f"")
        click.echo(f"   # Compare with baseline")
        click.echo(f"   tinyinfra benchmark compare \\")
        click.echo(f"       --baseline {model} \\")
        click.echo(f"       --optimized {output_path}")
        
    except Exception as e:
        click.echo(f"\n❌ Quantization failed: {e}", err=True)
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
    click.echo("💻 SYSTEM INFO")
    click.echo(f"{'='*60}")
    click.echo(f"\n🐍 Python: {sys.version.split()[0]}")
    click.echo(f"🔥 PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        click.echo(f"\n✅ CUDA: {torch.cuda.get_device_name(0)}")
        click.echo(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        click.echo(f"   Compute: {torch.cuda.get_device_capability(0)}")
    else:
        click.echo(f"\n❌ CUDA: Not available")
    
    # Check quantization libraries
    click.echo(f"\n📦 Quantization Libraries:")
    
    try:
        import awq
        click.echo(f"   ✅ autoawq: installed")
    except ImportError:
        click.echo(f"   ❌ autoawq: Not installed")
        click.echo(f"      Install: uv pip install autoawq")
    
    try:
        import bitsandbytes
        click.echo(f"   ✅ bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        click.echo(f"   ❌ bitsandbytes: Not installed")
        click.echo(f"      Install: uv pip install bitsandbytes")
    
    click.echo()


if __name__ == '__main__':
    cli()