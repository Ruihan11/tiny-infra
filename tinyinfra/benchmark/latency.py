"""
Latency benchmark: measure response time
"""
import torch
import time
import numpy as np
from typing import Dict, Optional
from torch.profiler import profile, ProfilerActivity
from pathlib import Path
import json

class LatencyBenchmark:
    """Measure generation latency (response time)"""
    
    def __init__(self, model):
        """
        Args:
            model: Llama3Model instance
        """
        self.model = model
    
    def run(
        self,
        prompt: str = "Explain artificial intelligence",
        num_tokens: int = 128,
        num_runs: int = 100,
        warmup_runs: int = 5,
        enable_profiler: bool = False,
        profile_output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run latency benchmark
        
        Args:
            prompt: Input text
            num_tokens: Number of tokens to generate
            num_runs: Number of benchmark iterations
            warmup_runs: Number of warmup iterations
            enable_profiler: Enable PyTorch profiler (profiles 3 runs only)
            profile_output_dir: Directory to save profiler traces
            
        Returns:
            dict with latency metrics and optional profiling data
        """
        print(f"\n⚡ Latency Benchmark")
        print(f"   Tokens: {num_tokens}")
        print(f"   Runs: {num_runs} (+ {warmup_runs} warmup)")
        if enable_profiler:
            print(f"   📊 Profiling: 3 runs only (lightweight)")
        
        # Measure First Token Latency (TTFT)
        print(f"\n⏱️  Measuring First Token Latency...")
        first_token_results = self._measure_first_token(
            prompt=prompt,
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        
        # Measure End-to-End Latency
        print(f"\n⏱️  Measuring End-to-End Latency...")
        e2e_results = self._measure_end_to_end(
            prompt=prompt,
            num_tokens=num_tokens,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            enable_profiler=enable_profiler,
            profile_output_dir=profile_output_dir
        )
        
        per_token_ms = e2e_results['mean_ms'] / num_tokens
        
        return {
            "first_token": first_token_results,
            "end_to_end": e2e_results,
            "per_token_ms": per_token_ms,
            "profiling_enabled": enable_profiler,
            "config": {
                "prompt_length": len(self.model.tokenizer.encode(prompt)),
                "num_tokens": num_tokens,
                "num_runs": num_runs
            }
        }
    
    def _measure_first_token(
        self,
        prompt: str,
        num_runs: int,
        warmup_runs: int
    ) -> Dict:
        """Measure time to first token (TTFT)"""
        inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        latencies = []
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model.model.generate(
                    inputs.input_ids,
                    max_new_tokens=1,
                    pad_token_id=self.model.tokenizer.pad_token_id
                )
        
        # Benchmark
        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model.model.generate(
                    inputs.input_ids,
                    max_new_tokens=1,
                    pad_token_id=self.model.tokenizer.pad_token_id
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            
            if (i + 1) % 20 == 0:
                print(f"      Progress: {i+1}/{num_runs}")
        
        return self._compute_stats(latencies, "First Token")
    
    def _measure_end_to_end(
        self,
        prompt: str,
        num_tokens: int,
        num_runs: int,
        warmup_runs: int,
        enable_profiler: bool,
        profile_output_dir: Optional[str]
    ) -> Dict:
        """Measure end-to-end generation latency"""
        latencies = []
        
        # Warmup
        for _ in range(warmup_runs):
            _ = self.model.generate(prompt, max_new_tokens=num_tokens)
        
        # Profile setup (only if enabled)
        prof = None
        if enable_profiler:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            
            prof = profile(
                activities=activities,
                record_shapes=False,   # Minimal config
                profile_memory=False,
                with_stack=False,
                with_flops=False,
            )
            prof.__enter__()
        
        # Benchmark
        profile_runs = 3 if enable_profiler else 0
        for i in range(num_runs):
            inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model.model.generate(
                    inputs.input_ids,
                    max_new_tokens=num_tokens,
                    pad_token_id=self.model.tokenizer.pad_token_id
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            
            # Stop profiling after 3 runs
            if enable_profiler and i == profile_runs - 1:
                prof.__exit__(None, None, None)
                profile_data = self._save_profile(prof, profile_output_dir)
            
            if (i + 1) % 10 == 0:
                print(f"      Progress: {i+1}/{num_runs}")
        
        stats = self._compute_stats(latencies, "End-to-End")
        
        if enable_profiler and profile_data:
            stats.update(profile_data)
        
        return stats
    
    def _save_profile(self, prof, output_dir: Optional[str]) -> Dict:
        """Save profile and return summary"""
        output_path = Path(output_dir or "results/profiling")
        output_path.mkdir(parents=True, exist_ok=True)
        
        trace_file = output_path / "latency_profile.json"
        prof.export_chrome_trace(str(trace_file))
        
        file_size_mb = trace_file.stat().st_size / (1024 * 1024)
        print(f"\n   📊 Profile saved: {trace_file} ({file_size_mb:.1f} MB)")
        print(f"      View in: chrome://tracing")
        
        # Top operations
        key_averages = prof.key_averages()
        sorted_ops = sorted(
            key_averages,
            key=lambda x: x.self_cuda_time_total if torch.cuda.is_available() else x.self_cpu_time_total,
            reverse=True
        )[:10]
        
        print(f"\n   📊 Top 5 Operations:")
        top_ops = []
        for i, op in enumerate(sorted_ops[:5]):
            time_ms = (op.self_cuda_time_total if torch.cuda.is_available() 
                      else op.self_cpu_time_total) / 1000
            print(f"      {i+1}. {op.key[:35]:<35} {time_ms:>8.2f} ms")
            
            top_ops.append({
                "name": op.key,
                "time_ms": float(time_ms)
            })
        
        summary = {
            "top_operations": top_ops,
            "file_size_mb": file_size_mb
        }
        
        return {
            "profile_trace": str(trace_file),
            "profile_summary": summary
        }
    
    def _compute_stats(self, latencies: list, name: str) -> Dict:
        """Compute statistical metrics"""
        latencies = np.array(latencies)
        
        return {
            "name": name,
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
    
    def print_results(self, results: Dict):
        """Pretty print results"""
        print(f"\n{'='*60}")
        print(f"⚡ LATENCY RESULTS")
        print(f"{'='*60}")
        
        ft = results['first_token']
        print(f"\n📍 First Token Latency (TTFT):")
        print(f"   Mean:   {ft['mean_ms']:.2f} ms")
        print(f"   Median: {ft['median_ms']:.2f} ms")
        print(f"   P95:    {ft['p95_ms']:.2f} ms")
        print(f"   P99:    {ft['p99_ms']:.2f} ms")
        
        e2e = results['end_to_end']
        print(f"\n📊 End-to-End Latency:")
        print(f"   Mean:   {e2e['mean_ms']:.2f} ms ({e2e['mean_ms']/1000:.2f} s)")
        print(f"   Median: {e2e['median_ms']:.2f} ms")
        print(f"   P95:    {e2e['p95_ms']:.2f} ms")
        print(f"   P99:    {e2e['p99_ms']:.2f} ms")
        
        print(f"\n⏱️  Per-Token Latency:")
        print(f"   {results['per_token_ms']:.2f} ms/token")
        
        if results.get('profiling_enabled'):
            e2e_summary = e2e.get('profile_summary', {})
            print(f"\n📊 Profiling:")
            print(f"   Trace:     {e2e.get('profile_trace', 'N/A')}")
            print(f"   File size: {e2e_summary.get('file_size_mb', 0):.1f} MB")
        
        print(f"\n{'='*60}\n")