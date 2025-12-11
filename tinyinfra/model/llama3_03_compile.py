"""
Torch.compile-Optimized Llama3 inference implementation
Built on top of llama3_02_opt.py with torch.compile for maximum performance

Key optimizations:
1. torch.compile with aggressive optimization modes for forward and decode
2. Compiled graph execution to reduce Python overhead
3. Dynamic shape compilation for flexible batch sizes
4. All existing optimizations from llama3_02_opt.py (fused ops, KV cache, FlashAttention)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import math


# ============================================================================
# Optimized Components for torch.compile
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm optimized for torch.compile"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RoPE(nn.Module):
    """RoPE optimized for torch.compile"""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 500000.0, device: str = "cuda"):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute and cache
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device: str = "cuda"):
        """Build cos/sin cache"""
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)

        cos = freqs.cos()
        sin = freqs.sin()

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to Q and K"""
        seq_len = q.shape[2]

        if start_pos + seq_len > self.max_seq_len:
            self._build_cache(start_pos + seq_len, self.inv_freq.device)
            self.max_seq_len = start_pos + seq_len

        cos = self.cos_cached[start_pos:start_pos + seq_len].to(q.dtype)
        sin = self.sin_cached[start_pos:start_pos + seq_len].to(q.dtype)

        q1, q2 = q[..., :q.shape[-1]//2], q[..., q.shape[-1]//2:]
        k1, k2 = k[..., :k.shape[-1]//2], k[..., k.shape[-1]//2:]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        return q_rot, k_rot


class SwiGLU(nn.Module):
    """SwiGLU optimized for torch.compile"""
    def __init__(self):
        super().__init__()

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up


# ============================================================================
# Compiled Components (using torch.compile)
# ============================================================================

class CompiledAttention(nn.Module):
    """
    Attention with torch.compile optimization
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 4096,
        rope_base: float = 500000.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Fused QKV projection
        self.qkv_proj = nn.Linear(
            dim,
            (n_heads + 2 * n_kv_heads) * self.head_dim,
            bias=False
        )
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # RoPE
        self.rotary_emb = RoPE(self.head_dim, max_seq_len, rope_base, device)

        # KV cache
        self.kv_cache: Optional[dict] = None

        # GQA expansion indices
        if self.n_rep > 1:
            self.register_buffer(
                "gqa_indices",
                torch.arange(n_kv_heads, device=device).repeat_interleave(self.n_rep),
                persistent=False
            )

    def _init_cache(self, batch_size: int):
        """Initialize KV cache"""
        self.kv_cache = {
            'k': torch.zeros(
                (batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim),
                device=self.device, dtype=self.dtype
            ),
            'v': torch.zeros(
                (batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim),
                device=self.device, dtype=self.dtype
            ),
            'current_len': 0
        }

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Fused QKV projection
        qkv = self.qkv_proj(x)
        
        # Split into Q, K, V
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        
        q = qkv[:, :, :q_size]
        k = qkv[:, :, q_size:q_size + kv_size]
        v = qkv[:, :, q_size + kv_size:]
        
        # Reshape
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE with Triton
        q, k = self.rotary_emb(q, k, start_pos)
        
        # KV cache
        if use_cache:
            if self.kv_cache is None or self.kv_cache['k'].shape[0] != batch_size:
                self._init_cache(batch_size)
            
            self.kv_cache['k'][:, :, start_pos:start_pos + seq_len] = k
            self.kv_cache['v'][:, :, start_pos:start_pos + seq_len] = v
            
            end_pos = start_pos + seq_len
            k = self.kv_cache['k'][:, :, :end_pos]
            v = self.kv_cache['v'][:, :, :end_pos]
        
        # GQA expansion
        if self.n_rep > 1:
            k = k.index_select(1, self.gqa_indices)
            v = v.index_select(1, self.gqa_indices)
        
        # FlashAttention via SDPA
        is_causal = (seq_len > 1) and not use_cache
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal
        )
        
        # Output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)

    def clear_cache(self):
        """Clear KV cache"""
        self.kv_cache = None


class CompiledFFN(nn.Module):
    """FFN with torch.compile"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.swiglu = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(self.swiglu(gate, up))


class CompiledTransformerBlock(nn.Module):
    """Transformer block with torch.compile"""
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_dim: int,
        max_seq_len: int = 4096,
        rope_base: float = 500000.0,
        rms_norm_eps: float = 1e-6,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.attention = CompiledAttention(
            dim, n_heads, n_kv_heads, max_seq_len, rope_base, device, dtype
        )
        self.feed_forward = CompiledFFN(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, rms_norm_eps)
        self.ffn_norm = RMSNorm(dim, rms_norm_eps)

    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int = 0, 
        use_cache: bool = False
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), start_pos, use_cache)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


# ============================================================================
# Main Model Class
# ============================================================================

class Llama3Compiled:
    """
    Torch.compile-optimized Llama3 for maximum performance

    Optimizations over llama3_02_opt.py:
    - torch.compile for forward pass (prefill and decode)
    - Reduced Python overhead via graph compilation
    - Dynamic shape support for flexible batching
    - All existing optimizations: fused QKV/gate-up, KV cache, FlashAttention
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_seq_len: int = 4096,
        compile_mode: str = "reduce-overhead",  # "default", "reduce-overhead", "max-autotune"
    ):
        """
        Initialize torch.compile-optimized Llama3 model

        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
            dtype: torch.float16 or torch.bfloat16
            max_seq_len: Maximum sequence length
            compile_mode: torch.compile mode
                - "default": Standard compilation
                - "reduce-overhead": Minimize Python overhead (recommended)
                - "max-autotune": Maximum optimization (slower compile, faster runtime)
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype if device == "cuda" else torch.float32
        self.max_seq_len = max_seq_len
        self.compile_mode = compile_mode

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load config
        print(f"⏳ Loading config from {model_name}...")
        config = AutoConfig.from_pretrained(model_name)

        self.vocab_size = config.vocab_size
        self.dim = config.hidden_size
        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = getattr(config, 'num_key_value_heads', self.n_heads)
        self.hidden_dim = config.intermediate_size
        self.rope_base = getattr(config, 'rope_theta', 500000.0)
        self.rms_norm_eps = getattr(config, 'rms_norm_eps', 1e-6)
        self.config = config

        print(f"   Config: dim={self.dim}, layers={self.n_layers}, heads={self.n_heads}")
        print(f"   Compile mode: {compile_mode}")

        # Build model
        self._build_model()
        
        # Load weights
        print(f"⏳ Loading weights...")
        self._load_weights(model_name)
        
        # Apply optimizations
        if device == "cuda":
            self._apply_cuda_optimizations()
        
        # Compile the model
        print(f"⏳ Compiling model with torch.compile (mode={compile_mode})...")
        self._compile_model()
        
        # Compatibility
        self.model = self
        self.use_flash_attn = True
        
        print(f"✅ Model loaded and compiled successfully!")
        print(f"   Memory: {self.get_memory_usage():.2f} GB")

    def _build_model(self):
        """Build model architecture"""
        self.embedding = nn.Embedding(self.vocab_size, self.dim)

        self.layers = nn.ModuleList([
            CompiledTransformerBlock(
                self.dim, self.n_heads, self.n_kv_heads, self.hidden_dim,
                self.max_seq_len, self.rope_base, self.rms_norm_eps,
                self.device, self.dtype
            )
            for _ in range(self.n_layers)
        ])

        self.norm = RMSNorm(self.dim, self.rms_norm_eps)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)
        
        # Move to device
        self.embedding = self.embedding.to(self.device, self.dtype)
        for layer in self.layers:
            layer.to(self.device, self.dtype)
        self.norm = self.norm.to(self.device, self.dtype)
        self.output = self.output.to(self.device, self.dtype)

    def _load_weights(self, model_name: str):
        """Load weights from HuggingFace model"""
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        hf_state = hf_model.state_dict()
        
        # Embedding
        self.embedding.weight.data.copy_(hf_state['model.embed_tokens.weight'])
        
        # Layers
        for i, layer in enumerate(self.layers):
            prefix = f'model.layers.{i}.'
            
            # Fused QKV
            q_weight = hf_state[f'{prefix}self_attn.q_proj.weight']
            k_weight = hf_state[f'{prefix}self_attn.k_proj.weight']
            v_weight = hf_state[f'{prefix}self_attn.v_proj.weight']
            layer.attention.qkv_proj.weight.data.copy_(
                torch.cat([q_weight, k_weight, v_weight], dim=0)
            )
            
            layer.attention.o_proj.weight.data.copy_(
                hf_state[f'{prefix}self_attn.o_proj.weight']
            )
            
            # Fused gate/up
            gate_weight = hf_state[f'{prefix}mlp.gate_proj.weight']
            up_weight = hf_state[f'{prefix}mlp.up_proj.weight']
            layer.feed_forward.gate_up_proj.weight.data.copy_(
                torch.cat([gate_weight, up_weight], dim=0)
            )
            
            layer.feed_forward.down_proj.weight.data.copy_(
                hf_state[f'{prefix}mlp.down_proj.weight']
            )
            
            # Norms
            layer.attention_norm.weight.data.copy_(
                hf_state[f'{prefix}input_layernorm.weight']
            )
            layer.ffn_norm.weight.data.copy_(
                hf_state[f'{prefix}post_attention_layernorm.weight']
            )
        
        # Final norm and output
        self.norm.weight.data.copy_(hf_state['model.norm.weight'])

        if 'lm_head.weight' in hf_state:
            self.output.weight.data.copy_(hf_state['lm_head.weight'])
        else:
            self.output.weight.data.copy_(hf_state['model.embed_tokens.weight'])
        
        # Cleanup
        del hf_model, hf_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _apply_cuda_optimizations(self):
        """Apply CUDA-specific optimizations"""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    def _compile_model(self):
        """Compile model with torch.compile in full activation mode"""
        # Configure dynamo for compilation
        import torch._dynamo
        torch._dynamo.config.suppress_errors = False  # Show errors for debugging
        torch._dynamo.config.verbose = False

        # Full activation mode: aggressive optimization settings
        torch._dynamo.config.cache_size_limit = 128

        # Compile forward pass with full graph optimization
        self.forward_compiled = torch.compile(
            self._forward_impl,
            mode=self.compile_mode,
            fullgraph=False,  # Allow fallback to eager mode for dynamic ops
            dynamic=True,     # Support dynamic shapes for different batch sizes
        )

        # Compile decode step with full graph optimization
        self.decode_compiled = torch.compile(
            self._decode_step_impl,
            mode=self.compile_mode,
            fullgraph=False,  # Allow fallback to eager mode for dynamic ops
            dynamic=True,     # Support dynamic shapes for different batch sizes
        )

        print(f"   ✅ Compiled forward and decode functions")

        # Warmup compilation with dummy inputs
        if self.device == "cuda":
            print(f"   ⏳ Warming up compiled functions...")
            self._warmup_compilation()
            print(f"   ✅ Warmup complete")

    def _warmup_compilation(self):
        """Warmup compiled functions with dummy inputs to trigger compilation"""
        try:
            with torch.inference_mode():
                # Mark step begin for CUDA graphs compatibility
                if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                    torch.compiler.cudagraph_mark_step_begin()

                # Warmup prefill with a small sequence
                dummy_tokens = torch.randint(
                    0, self.vocab_size, (1, 8), device=self.device, dtype=torch.long
                )
                out1 = self.forward_compiled(dummy_tokens, start_pos=0, use_cache=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                del out1

                # Mark step for next run
                if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                    torch.compiler.cudagraph_mark_step_begin()

                # Warmup decode with cache
                self.clear_cache()
                out2 = self.forward_compiled(dummy_tokens, start_pos=0, use_cache=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                del out2

                # Mark step for decode
                if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                    torch.compiler.cudagraph_mark_step_begin()

                dummy_token = torch.randint(
                    0, self.vocab_size, (1, 1), device=self.device, dtype=torch.long
                )
                out3 = self.decode_compiled(dummy_token, start_pos=8)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                del out3

                # Clear cache after warmup
                self.clear_cache()
        except Exception as e:
            # If warmup fails, just warn and continue - model will compile on first use
            print(f"   ⚠️  Warmup warning: {str(e)[:80]}... (will compile on first use)")

    def _forward_impl(
        self, 
        tokens: torch.Tensor, 
        start_pos: int = 0, 
        use_cache: bool = False
    ) -> torch.Tensor:
        """Forward pass implementation (will be compiled)"""
        x = self.embedding(tokens)
        
        for layer in self.layers:
            x = layer(x, start_pos, use_cache)
        
        x = self.norm(x)
        logits = self.output(x)

        return logits

    def _decode_step_impl(
        self, 
        token: torch.Tensor, 
        start_pos: int
    ) -> torch.Tensor:
        """Single decode step implementation (will be compiled)"""
        x = self.embedding(token)
        
        for layer in self.layers:
            x = layer(x, start_pos, use_cache=True)

        x = self.norm(x)
        return self.output(x)

    def forward(
        self, 
        tokens: torch.Tensor, 
        start_pos: int = 0, 
        use_cache: bool = False
    ) -> torch.Tensor:
        """Forward pass (uses compiled version)"""
        return self.forward_compiled(tokens, start_pos, use_cache)

    def clear_cache(self):
        """Clear KV cache"""
        for layer in self.layers:
            layer.attention.clear_cache()

    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        prompt: str = None,
        attention_mask=None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        pad_token_id=None,
        eos_token_id=None,
        use_cache: bool = True,
        **kwargs
    ):
        """
        Optimized text generation with compiled forward pass
        """
        # Handle input
        return_text = False
        if isinstance(input_ids, str):
            prompt = input_ids
            input_ids = None

        if input_ids is not None:
            tokens = input_ids.to(self.device)
        elif prompt is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            tokens = inputs.input_ids.to(self.device)
            return_text = True
        else:
            raise ValueError("Either input_ids or prompt must be provided")

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        batch_size = tokens.shape[0]
        
        if use_cache:
            self.clear_cache()

        # Pre-allocate output
        max_len = min(tokens.shape[1] + max_new_tokens, self.max_seq_len)
        output_tokens = torch.zeros(
            (batch_size, max_len), 
            dtype=torch.long, 
            device=self.device
        )
        output_tokens[:, :tokens.shape[1]] = tokens
        cur_len = tokens.shape[1]
        
        # Prefill with compiled forward
        logits = self.forward(tokens, start_pos=0, use_cache=use_cache)
        next_token_logits = logits[:, -1, :]
        cur_pos = tokens.shape[1]

        # Decode loop with compiled decode step
        for i in range(max_new_tokens):
            # Sample next token
            if do_sample and temperature > 0:
                next_token_logits = next_token_logits / temperature
                
                if top_k > 0:
                    top_k_vals, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits = torch.where(
                        next_token_logits < top_k_vals[:, -1:],
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumsum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
                    next_token_logits = sorted_logits.gather(-1, sorted_indices.argsort(-1))
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            output_tokens[:, cur_len] = next_token.squeeze(-1)
            cur_len += 1
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            
            if cur_len >= max_len:
                break
            
            # Compiled decode step
            if use_cache:
                next_token_logits = self.decode_compiled(next_token, cur_pos)[:, -1, :]
                cur_pos += 1
            else:
                logits = self.forward(output_tokens[:, :cur_len], start_pos=0, use_cache=False)
                next_token_logits = logits[:, -1, :]

        if use_cache:
            self.clear_cache()

        result_tokens = output_tokens[:, :cur_len]
        if return_text:
            return self.tokenizer.decode(result_tokens[0], skip_special_tokens=True)
        return result_tokens

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> List[str]:
        """Optimized batch generation"""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len - max_new_tokens
        )
        
        input_ids = inputs.input_ids.to(self.device)
        
        output_ids = self.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            use_cache=True
        )
        
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def get_memory_usage(self) -> float:
        """Get GPU memory usage in GB"""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return 0.0

    def get_model_size(self) -> float:
        """Get model size in GB"""
        param_size = sum(
            p.numel() * p.element_size()
            for p in self.embedding.parameters()
        )
        for layer in self.layers:
            param_size += sum(p.numel() * p.element_size() for p in layer.parameters())
        param_size += sum(p.numel() * p.element_size() for p in self.norm.parameters())
        param_size += sum(p.numel() * p.element_size() for p in self.output.parameters())
        return param_size / (1024 ** 3)