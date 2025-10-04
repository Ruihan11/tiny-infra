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
from .model.vllm.llama3_vllm import Llama3VLLM
from .benchmark.throughput import ThroughputBenchmark
from .benchmark.latency import LatencyBenchmark
from .benchmark.accuracy import AccuracyBenchmark


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
@click.option('--wrapper', default='hf', type=click.Choice(['hf', 'customized', 'vllm']), help='Model wrapper to use')
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

        # Use specific wrapper
        tinyinfra benchmark throughput --model meta-llama/Meta-Llama-3-8B --wrapper hf

        # Test quantized model
        tinyinfra benchmark throughput --model models/quantized/Meta-Llama-3-8B-awq-int4

        # Batch processing
        tinyinfra benchmark throughput \\
            --model models/quantized/Meta-Llama-3-8B-awq-int4 \\
            --batch-size 8 \\
            --num-tokens 256 \\
            --num-runs 50

        # Custom prompt
        tinyinfra benchmark throughput \\
            --model models/quantized/Meta-Llama-3-8B-awq-int4 \\
            --prompt "Write a Python function to" \\
            --num-tokens 512

        # With profiling
        tinyinfra benchmark throughput \\
            --model models/quantized/Meta-Llama-3-8B-awq-int4 \\
            --profile \\
            --output results/awq_profile

        # CPU inference
        tinyinfra benchmark throughput \\
            --model models/quantized/Meta-Llama-3-8B-awq-int4 \\
            --device cpu

    \b
    Wrappers:
        hf          - HuggingFace-based wrapper (default)
        customized  - Custom implementation wrapper (in development)
        vllm        - vLLM high-performance inference engine
    """
    click.echo(f"\n{'='*60}")
    click.echo("🚀 THROUGHPUT BENCHMARK")
    click.echo(f"{'='*60}")
    
    click.echo(f"\n📋 Configuration:")
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
    click.echo("\n⏳ Loading model...")
    try:
        # Handle device setting
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Select wrapper
        if wrapper == 'hf':
            model_instance = Llama3HF(model_name=model, device=device)
        elif wrapper == 'customized':
            model_instance = Llama3Customized(model_name=model, device=device)
        elif wrapper == 'vllm':
            model_instance = Llama3VLLM(model_name=model, device=device)
        else:
            raise ValueError(f"Unknown wrapper: {wrapper}")

        click.echo(f"✅ Loaded: {model_instance.get_model_size():.1f}GB")
        if device == 'cuda':
            click.echo(f"   Memory: {model_instance.get_memory_usage():.1f}GB")
    except Exception as e:
        click.echo(f"❌ Failed: {e}", err=True)
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
        
        click.echo(f"💾 Saved: {output}/throughput.json")
        
    except Exception as e:
        click.echo(f"❌ Failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@benchmark.command()
@click.option('--model', default='meta-llama/Meta-Llama-3-8B', help='Model name or path')
@click.option('--wrapper', default='hf', type=click.Choice(['hf', 'customized', 'vllm']), help='Model wrapper to use')
@click.option('--num-tokens', default=128, type=int, help='Number of tokens to generate')
@click.option('--num-runs', default=100, type=int, help='Number of benchmark runs')
@click.option('--warmup', default=5, type=int, help='Number of warmup runs')
@click.option('--prompt', default='Explain artificial intelligence', help='Input prompt for generation')
@click.option('--output', default='results/baseline', help='Output directory')
@click.option('--profile', is_flag=True, help='Enable profiling (lightweight)')
@click.option('--device', default='cuda', type=click.Choice(['cuda', 'cpu', 'auto']), help='Device to use')
def latency(model, wrapper, num_tokens, num_runs, warmup, prompt, output, profile, device):
    """
    Measure latency (response time)

    \b
    Examples:
        # Basic usage (default HF wrapper)
        tinyinfra benchmark latency --model meta-llama/Meta-Llama-3-8B

        # Use specific wrapper
        tinyinfra benchmark latency --model meta-llama/Meta-Llama-3-8B --wrapper hf

        # Test quantized model
        tinyinfra benchmark latency --model models/quantized/Meta-Llama-3-8B-awq-int4

        # More runs for accuracy
        tinyinfra benchmark latency \\
            --model models/quantized/Meta-Llama-3-8B-awq-int4 \\
            --num-runs 200 \\
            --num-tokens 256

        # Custom prompt
        tinyinfra benchmark latency \\
            --model models/quantized/Meta-Llama-3-8B-awq-int4 \\
            --prompt "Translate to French:" \\
            --num-tokens 100

        # With profiling
        tinyinfra benchmark latency \\
            --model models/quantized/Meta-Llama-3-8B-awq-int4 \\
            --profile \\
            --output results/awq_latency_profile

    \b
    Wrappers:
        hf          - HuggingFace-based wrapper (default)
        customized  - Custom implementation wrapper (in development)
        vllm        - vLLM high-performance inference engine
    """
    click.echo(f"\n{'='*60}")
    click.echo("⚡ LATENCY BENCHMARK")
    click.echo(f"{'='*60}")
    
    click.echo(f"\n📋 Configuration:")
    click.echo(f"   Model:      {model}")
    click.echo(f"   Wrapper:    {wrapper}")
    click.echo(f"   Device:     {device}")
    click.echo(f"   Tokens:     {num_tokens}")
    click.echo(f"   Runs:       {num_runs}")
    click.echo(f"   Warmup:     {warmup}")
    click.echo(f"   Prompt:     {prompt[:50]}...")
    click.echo(f"   Profiling:  {'YES' if profile else 'NO'}")

    # Load model
    click.echo("\n⏳ Loading model...")
    try:
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Select wrapper
        if wrapper == 'hf':
            model_instance = Llama3HF(model_name=model, device=device)
        elif wrapper == 'customized':
            model_instance = Llama3Customized(model_name=model, device=device)
        elif wrapper == 'vllm':
            model_instance = Llama3VLLM(model_name=model, device=device)
        else:
            raise ValueError(f"Unknown wrapper: {wrapper}")

        click.echo(f"✅ Loaded: {model_instance.get_model_size():.1f}GB")
        if device == 'cuda':
            click.echo(f"   Memory: {model_instance.get_memory_usage():.1f}GB")
    except Exception as e:
        click.echo(f"❌ Failed: {e}", err=True)
        sys.exit(1)
    
    # Benchmark
    try:
        bench = LatencyBenchmark(model_instance)
        results = bench.run(
            prompt=prompt,
            num_tokens=num_tokens,
            num_runs=num_runs,
            warmup_runs=warmup,
            enable_profiler=profile,
            profile_output_dir=output if profile else None
        )
        
        bench.print_results(results)
        
        # Save
        Path(output).mkdir(parents=True, exist_ok=True)
        with open(Path(output) / "latency.json", 'w') as f:
            results['config'] = {
                'model': model,
                'wrapper': wrapper,
                'device': device,
                'num_tokens': num_tokens,
                'num_runs': num_runs,
                'prompt': prompt
            }
            json.dump(results, f, indent=2)
        
        click.echo(f"💾 Saved: {output}/latency.json")
        
    except Exception as e:
        click.echo(f"❌ Failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@benchmark.command()
@click.option('--model', required=True, help='Model name or path')
@click.option('--wrapper', default='hf', type=click.Choice(['hf', 'customized', 'vllm']), help='Model wrapper to use')
@click.option('--split', default='validation', type=click.Choice(['validation', 'test']), help='Dataset split')
@click.option('--num-samples', default=None, type=int, help='Number of samples (None = all)')
@click.option('--subjects', default=None, help='Comma-separated subjects (None = all)')
@click.option('--output', default='results/accuracy', help='Output directory')
@click.option('--device', default='cuda', type=click.Choice(['cuda', 'cpu', 'auto']), help='Device')
def accuracy(model, wrapper, split, num_samples, subjects, output, device):
    """
    Measure model accuracy on MMLU dataset

    \b
    Examples:
        # Quick test (validation set, ~20 min, default HF wrapper)
        tinyinfra benchmark accuracy --model meta-llama/Meta-Llama-3-8B

        # Use specific wrapper
        tinyinfra benchmark accuracy --model meta-llama/Meta-Llama-3-8B --wrapper hf

        # Test quantized model
        tinyinfra benchmark accuracy --model models/quantized/Meta-Llama-3-8B-awq-int4

        # Sample 1000 questions for faster testing
        tinyinfra benchmark accuracy \\
            --model models/quantized/Meta-Llama-3-8B-awq-int4 \\
            --num-samples 1000

        # Test specific subjects
        tinyinfra benchmark accuracy \\
            --model meta-llama/Meta-Llama-3-8B \\
            --subjects "abstract_algebra,astronomy,computer_security"

        # Full test set (slow, ~3 hours)
        tinyinfra benchmark accuracy \\
            --model meta-llama/Meta-Llama-3-8B \\
            --split test

    \b
    Notes:
        - validation: 1,540 questions (~20 min)
        - test: 14,083 questions (~3 hours)
        - Requires: pip install datasets

    \b
    Wrappers:
        hf          - HuggingFace-based wrapper (default)
        customized  - Custom implementation wrapper (in development)
        vllm        - vLLM high-performance inference engine
    """
    click.echo(f"\n{'='*60}")
    click.echo("📊 MMLU ACCURACY BENCHMARK")
    click.echo(f"{'='*60}")
    
    click.echo(f"\n📋 Configuration:")
    click.echo(f"   Model:      {model}")
    click.echo(f"   Wrapper:    {wrapper}")
    click.echo(f"   Device:     {device}")
    click.echo(f"   Split:      {split}")
    click.echo(f"   Samples:    {num_samples if num_samples else 'all'}")
    if subjects:
        click.echo(f"   Subjects:   {subjects}")

    # Check datasets library
    try:
        import datasets
    except ImportError:
        click.echo(f"\n❌ 'datasets' library not found!", err=True)
        click.echo(f"   Install: pip install datasets")
        sys.exit(1)

    # Load model
    click.echo("\n⏳ Loading model...")
    try:
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Select wrapper
        if wrapper == 'hf':
            model_instance = Llama3HF(model_name=model, device=device)
        elif wrapper == 'customized':
            model_instance = Llama3Customized(model_name=model, device=device)
        elif wrapper == 'vllm':
            model_instance = Llama3VLLM(model_name=model, device=device)
        else:
            raise ValueError(f"Unknown wrapper: {wrapper}")

        click.echo(f"✅ Loaded: {model_instance.get_model_size():.1f}GB")
        if device == 'cuda':
            click.echo(f"   Memory: {model_instance.get_memory_usage():.1f}GB")
    except Exception as e:
        click.echo(f"❌ Failed: {e}", err=True)
        sys.exit(1)
    
    # Parse subjects
    subject_list = None
    if subjects:
        subject_list = [s.strip() for s in subjects.split(',')]
    
    # Run benchmark
    try:
        bench = AccuracyBenchmark(model_instance)

        results = bench.run(
            split=split,
            num_samples=num_samples,
            subjects=subject_list
        )
        
        bench.print_results(results)
        
        # Save
        Path(output).mkdir(parents=True, exist_ok=True)
        with open(Path(output) / f"accuracy_{split}.json", 'w') as f:
            # Save simplified results
            save_results = {
                'overall_accuracy': results['overall_accuracy'],
                'correct': results['correct'],
                'total': results['total'],
                'split': split,
                'model': model,
                'wrapper': wrapper,
                'num_samples': num_samples,
                'top_10_subjects': {}
            }
            
            # Add top 10 subjects
            sorted_subjects = sorted(
                results['subject_accuracies'].items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )[:10]
            
            for subject, stats in sorted_subjects:
                save_results['top_10_subjects'][subject] = stats
            
            json.dump(save_results, f, indent=2)
        
        click.echo(f"💾 Saved: {output}/accuracy_{split}.json")
        
    except Exception as e:
        click.echo(f"❌ Failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@benchmark.command()
@click.option('--model', default='meta-llama/Meta-Llama-3-8B', help='Model name or path')
@click.option('--wrapper', default='hf', type=click.Choice(['hf', 'customized', 'vllm']), help='Model wrapper to use')
@click.option('--output', default='results/baseline', help='Output directory')
def all(model, wrapper, output):
    """
    Run all benchmarks (throughput + latency)

    \b
    Examples:
        # Run with default HF wrapper
        tinyinfra benchmark all

        # Run with specific wrapper
        tinyinfra benchmark all --wrapper hf

        # Run on quantized model
        tinyinfra benchmark all --model models/quantized/Meta-Llama-3-8B-awq-int4

    \b
    Wrappers:
        hf          - HuggingFace-based wrapper (default)
        customized  - Custom implementation wrapper (in development)
        vllm        - vLLM high-performance inference engine
    """
    click.echo(f"\n{'='*60}")
    click.echo("🎯 FULL BENCHMARK")
    click.echo(f"{'='*60}")

    try:
        # Select wrapper
        if wrapper == 'hf':
            model_instance = Llama3HF(model_name=model)
        elif wrapper == 'customized':
            model_instance = Llama3Customized(model_name=model)
        elif wrapper == 'vllm':
            model_instance = Llama3VLLM(model_name=model)
        else:
            raise ValueError(f"Unknown wrapper: {wrapper}")
        
        # Throughput
        click.echo("\n[1/2] Throughput...")
        bench_tp = ThroughputBenchmark(model_instance)
        res_tp = bench_tp.run(num_runs=20)
        bench_tp.print_results(res_tp)

        # Latency
        click.echo("\n[2/2] Latency...")
        bench_lat = LatencyBenchmark(model_instance)
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
    Quantize with AWQ (Activation-aware Weight Quantization)
    
    AWQ is recommended for 4-bit quantization with minimal accuracy loss.
    
    \b
    Examples:
        # 4-bit quantization (recommended)
        tinyinfra quantize awq --model meta-llama/Meta-Llama-3-8B --bits 4
        
        # 8-bit quantization
        tinyinfra quantize awq --model meta-llama/Meta-Llama-3-8B --bits 8
        
        # Custom output directory
        tinyinfra quantize awq \\
            --model meta-llama/Meta-Llama-3-8B \\
            --bits 4 \\
            --output my_models/quantized
        
        # Custom group size (default 128)
        tinyinfra quantize awq \\
            --model meta-llama/Meta-Llama-3-8B \\
            --bits 4 \\
            --group-size 64
    
    \b
    After quantization, test performance:
        tinyinfra benchmark throughput --model models/quantized/Meta-Llama-3-8B-awq-int4
        tinyinfra benchmark accuracy --model models/quantized/Meta-Llama-3-8B-awq-int4
    """
    from .quantization.quantizer import AWQQuantizer
    
    try:
        quantizer = AWQQuantizer(model)
        output_path = quantizer.quantize(
            bits=int(bits),
            output_dir=output,
            group_size=group_size
        )
        
        click.echo(f"\n" + "="*60)
        click.echo("✅ QUANTIZATION COMPLETE")
        click.echo("="*60)
        click.echo(f"\nQuantized model: {output_path}")
        
        click.echo(f"\n📋 Next Steps:")
        click.echo(f"\n1️⃣  Test performance:")
        click.echo(f"   tinyinfra benchmark throughput --model {output_path}")
        
        click.echo(f"\n2️⃣  Test accuracy:")
        click.echo(f"   tinyinfra benchmark accuracy --model {output_path}")
        
        click.echo(f"\n3️⃣  Test latency:")
        click.echo(f"   tinyinfra benchmark latency --model {output_path}")
        
        click.echo(f"\n" + "="*60 + "\n")
        
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
    
    \b
    Examples:
        # 8-bit quantization
        tinyinfra quantize bnb --model meta-llama/Meta-Llama-3-8B --bits 8
        
        # 4-bit quantization
        tinyinfra quantize bnb --model meta-llama/Meta-Llama-3-8B --bits 4
    
    \b
    After quantization, test performance:
        tinyinfra benchmark throughput --model models/quantized/Meta-Llama-3-8B-bnb-int8
        tinyinfra benchmark accuracy --model models/quantized/Meta-Llama-3-8B-bnb-int8
    """
    from .quantization.quantizer import BitsAndBytesQuantizer
    
    try:
        quantizer = BitsAndBytesQuantizer(model)
        output_path = quantizer.quantize(
            bits=int(bits),
            output_dir=output
        )
        
        click.echo(f"\n" + "="*60)
        click.echo("✅ QUANTIZATION COMPLETE")
        click.echo("="*60)
        click.echo(f"\nQuantized model: {output_path}")
        
        click.echo(f"\n📋 Next Steps:")
        click.echo(f"\n1️⃣  Test performance:")
        click.echo(f"   tinyinfra benchmark throughput --model {output_path}")
        
        click.echo(f"\n2️⃣  Test accuracy:")
        click.echo(f"   tinyinfra benchmark accuracy --model {output_path}")
        
        click.echo(f"\n3️⃣  Test latency:")
        click.echo(f"   tinyinfra benchmark latency --model {output_path}")
        
        click.echo(f"\n" + "="*60 + "\n")
        
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
        click.echo(f"      Install: pip install autoawq")
    
    try:
        import bitsandbytes
        click.echo(f"   ✅ bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        click.echo(f"   ❌ bitsandbytes: Not installed")
        click.echo(f"      Install: pip install bitsandbytes")
    
    # Check datasets
    click.echo(f"\n📚 Datasets:")
    try:
        import datasets
        click.echo(f"   ✅ datasets: {datasets.__version__}")
    except ImportError:
        click.echo(f"   ❌ datasets: Not installed")
        click.echo(f"      Install: pip install datasets")
    
    click.echo()


# ============================================================
# QUICKSTART COMMAND
# ============================================================

@cli.command()
@click.option('--model', default='meta-llama/Meta-Llama-3-8B', help='Model to use')
def quickstart(model):
    """
    Quick start guide with example commands
    
    Shows common workflows and example commands.
    """
    click.echo(f"\n{'='*70}")
    click.echo("🚀 TINYINFRA QUICK START")
    click.echo(f"{'='*70}")
    
    click.echo(f"\n📋 STEP 1: Check your system")
    click.echo(f"   tinyinfra info")
    
    click.echo(f"\n📋 STEP 2: Run baseline benchmark")
    click.echo(f"   tinyinfra benchmark throughput --model {model}")
    click.echo(f"   tinyinfra benchmark latency --model {model}")
    click.echo(f"   tinyinfra benchmark accuracy --model {model}")
    
    click.echo(f"\n📋 STEP 3: Quantize model (AWQ 4-bit recommended)")
    click.echo(f"   tinyinfra quantize awq --model {model} --bits 4")
    
    click.echo(f"\n📋 STEP 4: Test quantized model")
    model_name = Path(model).name
    quantized_path = f"models/quantized/{model_name}-awq-int4"
    click.echo(f"   tinyinfra benchmark throughput --model {quantized_path}")
    click.echo(f"   tinyinfra benchmark accuracy --model {quantized_path}")
    
    click.echo(f"\n{'='*70}")
    click.echo("📚 More Examples:")
    click.echo(f"{'='*70}")
    
    click.echo(f"\n🔧 Different quantization methods:")
    click.echo(f"   # AWQ (best for 4-bit)")
    click.echo(f"   tinyinfra quantize awq --model {model} --bits 4")
    click.echo(f"")
    click.echo(f"   # BitsAndBytes (simple, fast)")
    click.echo(f"   tinyinfra quantize bnb --model {model} --bits 8")
    
    click.echo(f"\n📊 Advanced profiling:")
    click.echo(f"   # With PyTorch profiler")
    click.echo(f"   tinyinfra benchmark throughput --model {model} --profile")
    click.echo(f"")
    click.echo(f"   # Custom parameters")
    click.echo(f"   tinyinfra benchmark throughput \\")
    click.echo(f"       --model {model} \\")
    click.echo(f"       --batch-size 8 \\")
    click.echo(f"       --num-runs 50")
    
    click.echo(f"\n📊 Accuracy testing:")
    click.echo(f"   # Quick validation (1,540 questions)")
    click.echo(f"   tinyinfra benchmark accuracy --model {quantized_path}")
    click.echo(f"")
    click.echo(f"   # Sample 1000 for faster testing")
    click.echo(f"   tinyinfra benchmark accuracy --model {quantized_path} --num-samples 1000")
    
    click.echo(f"\n💡 Tips:")
    click.echo(f"   - Use --help on any command for details")
    click.echo(f"   - AWQ 4-bit: Best balance of size/speed/accuracy")
    click.echo(f"   - BnB 8-bit: Safest option with minimal accuracy loss")
    click.echo(f"   - BnB 4-bit: Most aggressive compression")
    click.echo(f"   - Accuracy testing: validation set is usually sufficient")
    
    click.echo(f"\n{'='*70}\n")


if __name__ == '__main__':
    cli()