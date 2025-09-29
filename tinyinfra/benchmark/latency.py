"""
Latency benchmark: measure response time
"""
import torch
import time
import numpy as np
from typing import Dict

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
        num_runs: int = 20,
        warmup_runs: int = 5
    ) -> Dict:
        """
        Run latency benchmark
        
        Measures:
        - First token latency (TTFT)
        - Per-token latency
        - End-to-end latency
        
        Args:
            prompt: Input text
            num_tokens: Number of tokens to generate
            num_runs: Number of benchmark iterations
            warmup_runs: Number of warmup iterations
            
        Returns:
            dict with latency metrics
        """
        print(f"\n⚡ Latency Benchmark")
        print(f"   Tokens: {num_tokens}")
        print(f"   Runs: {num_runs} (+ {warmup_runs} warmup)")
        
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
            warmup_runs=warmup_runs
        )
        
        # Calculate per-token latency
        per_token_ms = e2e_results['mean_ms'] / num_tokens
        
        return {
            "first_token": first_token_results,
            "end_to_end": e2e_results,
            "per_token_ms": per_token_ms,
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
        
        # Tokenize once
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
        warmup_runs: int
    ) -> Dict:
        """Measure end-to-end generation latency"""
        
        latencies = []
        
        # Warmup
        for _ in range(warmup_runs):
            _ = self.model.generate(prompt, max_new_tokens=num_tokens)
        
        # Benchmark
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
            
            if (i + 1) % 10 == 0:
                print(f"      Progress: {i+1}/{num_runs}")
        
        return self._compute_stats(latencies, "End-to-End")
    
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
        
        # First Token
        ft = results['first_token']
        print(f"\n📍 First Token Latency (TTFT):")
        print(f"   Mean:   {ft['mean_ms']:.2f} ms")
        print(f"   Median: {ft['median_ms']:.2f} ms")
        print(f"   P95:    {ft['p95_ms']:.2f} ms")
        print(f"   P99:    {ft['p99_ms']:.2f} ms")
        
        # End-to-End
        e2e = results['end_to_end']
        print(f"\n📊 End-to-End Latency:")
        print(f"   Mean:   {e2e['mean_ms']:.2f} ms ({e2e['mean_ms']/1000:.2f} s)")
        print(f"   Median: {e2e['median_ms']:.2f} ms")
        print(f"   P95:    {e2e['p95_ms']:.2f} ms")
        print(f"   P99:    {e2e['p99_ms']:.2f} ms")
        
        # Per-token
        print(f"\n⏱️  Per-Token Latency:")
        print(f"   {results['per_token_ms']:.2f} ms/token")
        
        print(f"\n{'='*60}\n")