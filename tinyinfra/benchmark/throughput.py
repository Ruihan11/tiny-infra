"""
Throughput benchmark: measure tokens/second
"""
import torch
import time
from typing import Dict, List, Optional
from torch.profiler import profile, ProfilerActivity
from pathlib import Path
import json

class ThroughputBenchmark:
    """Measure generation throughput (tokens/second)"""
    
    def __init__(self, model):
        """
        Args:
            model: Llama3Model instance
        """
        self.model = model
    
    def run(
        self,
        prompt: str = "Explain artificial intelligence",
        batch_size: int = 1,
        num_tokens: int = 128,
        num_runs: int = 10,
        warmup_runs: int = 3,
        enable_profiler: bool = False,
        profile_output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run throughput benchmark
        
        Args:
            prompt: Input text
            batch_size: Number of prompts to process simultaneously
            num_tokens: Number of tokens to generate per prompt
            num_runs: Number of benchmark iterations
            warmup_runs: Number of warmup iterations (not counted)
            enable_profiler: Enable PyTorch profiler (profiles first 3 runs only)
            profile_output_dir: Directory to save profiler traces (if enabled)
            
        Returns:
            dict with throughput metrics and optional profiling data
        """
        print(f"\n🔥 Throughput Benchmark")
        print(f"   Batch size: {batch_size}")
        print(f"   Tokens per run: {num_tokens}")
        print(f"   Runs: {num_runs} (+ {warmup_runs} warmup)")
        if enable_profiler:
            print(f"   📊 Profiling: First 3 runs only (lightweight)")
        
        # Prepare prompts
        if batch_size == 1:
            prompts = [prompt]
        else:
            prompts = [prompt] * batch_size
        
        # Warmup
        print(f"\n⏳ Warming up ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            _ = self._generate_batch(prompts, num_tokens)
            print(f"   Warmup {i+1}/{warmup_runs}")
        
        # Benchmark
        print(f"\n⚡ Running benchmark ({num_runs} runs)...")
        times = []
        times_no_profiling = []  # Track non-profiled runs separately
        total_tokens = 0
        profile_data = None

        # Initialize profile_data_to_save to None to ensure it exists
        profile_data_to_save = None
        for i in range(num_runs):
            # Only profile first 3 runs to keep file size small
            should_profile = enable_profiler and i < 3

            if should_profile:
                run_time, prof = self._run_single_with_profiling(prompts, num_tokens, i)
                if i == 2:  # Save profile after 3rd run
                    # Store profile data to be processed later
                    profile_data_to_save = prof
            else:
                run_time = self._run_single(prompts, num_tokens)
                times_no_profiling.append(run_time)  # Only count non-profiled runs

            times.append(run_time)
            tokens_generated = num_tokens * batch_size
            total_tokens += tokens_generated

            throughput = tokens_generated / run_time
            print(f"   Run {i+1}/{num_runs}: {throughput:.2f} tok/s")

        # Process profiling data after all runs are complete
        if enable_profiler and profile_data_to_save is not None:
            profile_data = self._save_profile(profile_data_to_save, profile_output_dir)

        # Calculate results - use only non-profiled runs if profiling was enabled
        if enable_profiler and times_no_profiling:
            # Exclude profiled runs from throughput calculation
            total_time = sum(times_no_profiling)
            mean_time = total_time / len(times_no_profiling)
            total_tokens_no_profiling = num_tokens * batch_size * len(times_no_profiling)
            tokens_per_run = num_tokens * batch_size
        else:
            # No profiling - use all runs
            total_time = sum(times)
            mean_time = total_time / num_runs
            total_tokens_no_profiling = total_tokens
            tokens_per_run = num_tokens * batch_size
        
        results = {
            "throughput_tokens_per_sec": total_tokens_no_profiling / total_time,
            "mean_latency_sec": mean_time,
            "total_tokens": total_tokens,
            "total_time_sec": total_time,
            "tokens_per_run": tokens_per_run,
            "runs": num_runs,
            "runs_used_for_throughput": len(times_no_profiling) if enable_profiler else num_runs,
            "batch_size": batch_size,
            "profiling_enabled": enable_profiler
        }
        
        if profile_data:
            results.update(profile_data)
        
        return results
    
    def _run_single(self, prompts: List[str], num_tokens: int) -> float:
        """Run a single benchmark iteration without profiling"""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        
        _ = self._generate_batch(prompts, num_tokens)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return time.perf_counter() - start
    
    def _run_single_with_profiling(self, prompts: List[str], num_tokens: int, run_id: int):
        """Run a single iteration with profiling"""
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        # Minimal profiler config - keeps file size small
        prof = profile(
            activities=activities,
            record_shapes=False,      # ❌ Disable shapes (saves space)
            profile_memory=False,     # ❌ Disable memory profiling (saves space)
            with_stack=False,         # ❌ Disable stack traces (saves space)
            with_flops=False,         # ❌ Disable FLOPs calculation
        )
        
        prof.__enter__()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        
        _ = self._generate_batch(prompts, num_tokens)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        
        prof.__exit__(None, None, None)
        
        return elapsed, prof
    
    def _get_event_time(self, event):
        """Get time from event, handling both CPU and CUDA"""
        # Try CUDA time first, fall back to CPU time
        if hasattr(event, 'cuda_time_total') and event.cuda_time_total > 0:
            return event.cuda_time_total
        elif hasattr(event, 'self_cuda_time_total') and event.self_cuda_time_total > 0:
            return event.self_cuda_time_total
        else:
            return event.self_cpu_time_total
    
    def _save_profile(self, prof, output_dir: Optional[str]) -> Dict:
        """Save profile and return summary"""
        output_path = Path(output_dir or "results/profiling")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save minimal trace
        trace_file = output_path / "throughput_profile.json"
        prof.export_chrome_trace(str(trace_file))
        
        file_size_mb = trace_file.stat().st_size / (1024 * 1024)
        print(f"\n   📊 Profile saved: {trace_file} ({file_size_mb:.1f} MB)")
        print(f"      View in: https://ui.perfetto.dev/")
        
        # Get top operations - use helper function to get time
        key_averages = prof.key_averages()
        sorted_ops = sorted(
            key_averages,
            key=lambda x: self._get_event_time(x),
            reverse=True
        )[:10]
        
        print(f"\n   📊 Top 5 Operations:")
        top_ops = []
        for i, op in enumerate(sorted_ops[:5]):
            time_ms = self._get_event_time(op) / 1000  # Convert to ms
            print(f"      {i+1}. {op.key[:35]:<35} {time_ms:>8.2f} ms")
            
            top_ops.append({
                "name": op.key,
                "time_ms": float(time_ms),
                "calls": op.count
            })
        
        # Save summary
        summary = {
            "top_operations": top_ops,
            "trace_file": str(trace_file.name),
            "file_size_mb": file_size_mb
        }
        
        summary_file = output_path / "throughput_profile_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return {
            "profile_trace": str(trace_file),
            "profile_summary": summary
        }
    
    def _generate_batch(self, prompts: List[str], num_tokens: int) -> List[str]:
        """Generate for a batch of prompts"""
        # Check if model has native batch generation (e.g., vLLM)
        if hasattr(self.model, 'generate_batch'):
            return self.model.generate_batch(
                prompts,
                max_new_tokens=num_tokens,
                temperature=1.0
            )
        elif len(prompts) == 1:
            output = self.model.generate(
                prompts[0],
                max_new_tokens=num_tokens,
                temperature=1.0
            )
            return [output]
        else:
            # Use HuggingFace-style batch generation
            inputs = self.model.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=num_tokens,
                    pad_token_id=self.model.tokenizer.pad_token_id
                )

            return self.model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def print_results(self, results: Dict):
        """Pretty print results"""
        print(f"\n{'='*50}")
        print(f"📊 THROUGHPUT RESULTS")
        print(f"{'='*50}")
        print(f"Throughput:      {results['throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"Mean Latency:    {results['mean_latency_sec']*1000:.2f} ms")
        print(f"Total Tokens:    {results['total_tokens']:,}")
        print(f"Total Time:      {results['total_time_sec']:.2f} sec")
        print(f"Tokens/Run:      {results['tokens_per_run']}")
        print(f"Batch Size:      {results['batch_size']}")
        
        if results.get('profiling_enabled'):
            summary = results.get('profile_summary', {})
            print(f"\n📊 Profiling:")
            print(f"   Trace:     {results.get('profile_trace', 'N/A')}")
            print(f"   File size: {summary.get('file_size_mb', 0):.1f} MB")
        
        print(f"{'='*50}\n")