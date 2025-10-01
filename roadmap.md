# roadmap
Llama-3-8B End-to-End Inference Optimization
> Understand bottlenecks → Targeted optimization → Validate results  

Phase 1: Performance Profiling (Week 1-2)

- [x] Establish baseline metrics (latency, throughput, memory, accuracy)
- [x] PyTorch Profiler for operator-level analysis
- [ ] Nsight Systems for GPU kernel analysis (Maybe for future reference)
- [x] Custom vs HuggingFace implementation comparison

Phase 2: Model Quantization (Week 3-5)

- [x] AWQ INT4 quantization implementation
- [x] BitsAndBytes INT4/INT8 quantization
- [x] Performance benchmarking (A40 GPU)
- [x] Accuracy evaluation metrics
- [ ] Mixed precision strategies (sensitive layers in FP16)
- [ ] Per-channel vs per-tensor quantization comparison
- [ ] GPTQ quantization integration
- [ ] Quantization-aware training experiments

Phase 3: Custom Implementation Optimization (Week 6-7)

- [x] Separate HuggingFace and custom implementations
- [x] Pure PyTorch Llama3 with RoPE and GQA
- [ ] Implement KV cache for inference
- [ ] Flash Attention integration
- [ ] Fused kernel operations (RMSNorm + Attention)
- [ ] Memory-efficient attention patterns
- [ ] Benchmark custom vs HuggingFace

Phase 4: TensorRT Deployment (Week 8-9)

- [ ] PyTorch → ONNX → TensorRT conversion pipeline
- [ ] Handle dynamic shapes and KV cache
- [ ] FP16/INT8 TensorRT precision tuning
- [ ] TensorRT layer fusion optimization
- [ ] Profile with TensorRT Profiler
- [ ] Identify TensorRT limitations

Phase 5: Kernel Optimization (Week 10-11)

- [ ] Learn Triton programming (Vector Add → GEMM → Softmax)
- [ ] Flash Attention 2/3 integration
- [ ] Custom operator fusion (SwiGLU, RMSNorm, QKV projection)
- [ ] Roofline model analysis
- [ ] Performance tuning (BLOCK_SIZE, num_warps)
- [ ] Quantized GEMM kernels


---  

9/30/2025
- seperate pytorch infer v.s. hf infer
- runs well but not as fast

customized  
Throughput:      40.80 tokens/sec  
Mean Latency:    50195.74 ms  
Total Tokens:    40,960  
Total Time:      1003.91 sec  
Tokens/Run:      2048  
Batch Size:      8  

hf  
Throughput:      229.73 tokens/sec  
Mean Latency:    8914.88 ms  
Total Tokens:    40,960  
Total Time:      178.30 sec  
Tokens/Run:      2048  
Batch Size:      8  

---  

9/29/2025
- first commit
- full benchmark with throughput latency and mmlu accuracy


