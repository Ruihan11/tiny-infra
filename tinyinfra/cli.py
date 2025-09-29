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
from .benchmark.latency import LatencyBenchmark  # 新增


@click.group()
def cli():
    """
    🚀 TinyInfra: LLM Inference Optimization Toolkit
    """
    pass


@cli.group()
def benchmark():
    """Run benchmarks"""
    pass


@benchmark.command()
@click.option('--model', default='meta-llama/Meta-Llama-3-8B', 
              help='Model name')
@click.option('--device', default='cuda', 
              help='Device (cuda/cpu)')
@click.option('--batch-size', default=1, type=int,
              help='Batch size')
@click.option('--num-tokens', default=128, type=int,
              help='Number of tokens to generate')
@click.option('--num-runs', default=10, type=int,
              help='Number of benchmark runs')
@click.option('--warmup', default=3, type=int,
              help='Number of warmup runs')
@click.option('--output', default='results/baseline',
              help='Output directory for results')
@click.option('--prompt', default='Explain artificial intelligence',
              help='Input prompt for generation')
def throughput(model, device, batch_size, num_tokens, num_runs, warmup, output, prompt):
    """
    Measure throughput (tokens/second)
    
    Example:
        tinyinfra benchmark throughput
        tinyinfra benchmark throughput --batch-size 8
    """
    click.echo("\n" + "="*60)
    click.echo("🚀 TINYINFRA THROUGHPUT BENCHMARK")
    click.echo("="*60)
    
    # Configuration
    click.echo("\n📋 Configuration:")
    click.echo(f"   Model:      {model}")
    click.echo(f"   Device:     {device}")
    click.echo(f"   Batch size: {batch_size}")
    click.echo(f"   Tokens:     {num_tokens}")
    click.echo(f"   Runs:       {num_runs}")
    
    # Load model
    click.echo("\n⏳ Loading model...")
    try:
        llama = Llama3Model(model_name=model, device=device)
        click.echo(f"✅ Model loaded!")
        click.echo(f"   Size: {llama.get_model_size():.2f} GB")
        click.echo(f"   Memory: {llama.get_memory_usage():.2f} GB")
    except Exception as e:
        click.echo(f"❌ Failed to load model: {e}", err=True)
        sys.exit(1)
    
    # Run benchmark
    try:
        bench = ThroughputBenchmark(llama)
        results = bench.run(
            prompt=prompt,
            batch_size=batch_size,
            num_tokens=num_tokens,
            num_runs=num_runs,
            warmup_runs=warmup
        )
        
        bench.print_results(results)
        
        # Save results
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "throughput.json"
        with open(output_file, 'w') as f:
            save_results = {k: v for k, v in results.items() if k != 'all_times'}
            json.dump(save_results, f, indent=2)
        
        click.echo(f"💾 Results saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@benchmark.command()  # 新增 latency 命令
@click.option('--model', default='meta-llama/Meta-Llama-3-8B', 
              help='Model name')
@click.option('--device', default='cuda', 
              help='Device (cuda/cpu)')
@click.option('--num-tokens', default=128, type=int,
              help='Number of tokens to generate')
@click.option('--num-runs', default=100, type=int,
              help='Number of benchmark runs')
@click.option('--warmup', default=5, type=int,
              help='Number of warmup runs')
@click.option('--output', default='results/baseline',
              help='Output directory for results')
@click.option('--prompt', default='Explain artificial intelligence',
              help='Input prompt for generation')
def latency(model, device, num_tokens, num_runs, warmup, output, prompt):
    """
    Measure latency (response time)
    
    Measures:
    - First Token Latency (TTFT)
    - End-to-End Latency
    - Per-Token Latency
    
    Example:
        tinyinfra benchmark latency
        tinyinfra benchmark latency --num-runs 200
    """
    click.echo("\n" + "="*60)
    click.echo("⚡ TINYINFRA LATENCY BENCHMARK")
    click.echo("="*60)
    
    # Configuration
    click.echo("\n📋 Configuration:")
    click.echo(f"   Model:      {model}")
    click.echo(f"   Device:     {device}")
    click.echo(f"   Tokens:     {num_tokens}")
    click.echo(f"   Runs:       {num_runs}")
    
    # Load model
    click.echo("\n⏳ Loading model...")
    try:
        llama = Llama3Model(model_name=model, device=device)
        click.echo(f"✅ Model loaded!")
        click.echo(f"   Size: {llama.get_model_size():.2f} GB")
        click.echo(f"   Memory: {llama.get_memory_usage():.2f} GB")
    except Exception as e:
        click.echo(f"❌ Failed to load model: {e}", err=True)
        sys.exit(1)
    
    # Run benchmark
    try:
        bench = LatencyBenchmark(llama)
        results = bench.run(
            prompt=prompt,
            num_tokens=num_tokens,
            num_runs=num_runs,
            warmup_runs=warmup
        )
        
        bench.print_results(results)
        
        # Save results
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "latency.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        click.echo(f"💾 Results saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@benchmark.command()  # 新增 all 命令
@click.option('--model', default='meta-llama/Meta-Llama-3-8B', help='Model name')
@click.option('--device', default='cuda', help='Device (cuda/cpu)')
@click.option('--output', default='results/baseline', help='Output directory')
def all(model, device, output):
    """
    Run all benchmarks (throughput + latency)
    
    Example:
        tinyinfra benchmark all
    """
    click.echo("\n" + "="*60)
    click.echo("🎯 TINYINFRA FULL BENCHMARK SUITE")
    click.echo("="*60)
    
    # Load model once
    click.echo("\n⏳ Loading model...")
    try:
        llama = Llama3Model(model_name=model, device=device)
        click.echo(f"✅ Model loaded!")
    except Exception as e:
        click.echo(f"❌ Failed to load model: {e}", err=True)
        sys.exit(1)
    
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Throughput
    click.echo("\n" + "="*60)
    click.echo("📊 [1/2] Running Throughput Benchmark")
    click.echo("="*60)
    try:
        bench_tp = ThroughputBenchmark(llama)
        results_tp = bench_tp.run(num_runs=20)
        bench_tp.print_results(results_tp)
        
        with open(output_dir / "throughput.json", 'w') as f:
            save_results = {k: v for k, v in results_tp.items() if k != 'all_times'}
            json.dump(save_results, f, indent=2)
    except Exception as e:
        click.echo(f"❌ Throughput benchmark failed: {e}", err=True)
    
    # 2. Latency
    click.echo("\n" + "="*60)
    click.echo("⚡ [2/2] Running Latency Benchmark")
    click.echo("="*60)
    try:
        bench_lat = LatencyBenchmark(llama)
        results_lat = bench_lat.run(num_runs=100)
        bench_lat.print_results(results_lat)
        
        with open(output_dir / "latency.json", 'w') as f:
            json.dump(results_lat, f, indent=2)
    except Exception as e:
        click.echo(f"❌ Latency benchmark failed: {e}", err=True)
    
    click.echo("\n" + "="*60)
    click.echo("✅ ALL BENCHMARKS COMPLETE")
    click.echo("="*60)
    click.echo(f"\n💾 Results saved to: {output_dir}/")
    click.echo(f"   - throughput.json")
    click.echo(f"   - latency.json\n")


@cli.command()
def info():
    """Show system information"""
    click.echo("\n" + "="*60)
    click.echo("💻 SYSTEM INFORMATION")
    click.echo("="*60)
    
    click.echo(f"\n🐍 Python: {sys.version.split()[0]}")
    click.echo(f"🔥 PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        click.echo(f"\n✅ CUDA Available")
        click.echo(f"   Devices: {torch.cuda.device_count()}")
        click.echo(f"   Device: {torch.cuda.get_device_name(0)}")
        click.echo(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        click.echo(f"\n❌ CUDA Not Available (using CPU)")
    
    click.echo()


if __name__ == '__main__':
    cli()