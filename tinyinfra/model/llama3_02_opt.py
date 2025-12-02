"""
Pure PyTorch Llama3 inference implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor"""
    x1, x2 = x.chunk(2, dim=-1)
    d = x1.shape[-1]
    return torch.cat([
        x1 * cos[..., :d] - x2 * sin[..., :d],
        x2 * cos[..., :d] + x1 * sin[..., :d]
    ], dim=-1)


class Attention(nn.Module):
    """Multi-head attention with RoPE, Grouped Query Attention (GQA), KV cache, and FlashAttention"""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = None, max_seq_len: int = 2048, use_flash_attn: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

        # KV cache buffers
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        use_cache: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE with correct position offset
        cos, sin = self.rotary_emb(x, start_pos + seq_len)
        cos = cos[start_pos:start_pos + seq_len]
        sin = sin[start_pos:start_pos + seq_len]
        q = apply_rotary_emb(q, cos[None, None, :, :], sin[None, None, :, :])
        k = apply_rotary_emb(k, cos[None, None, :, :], sin[None, None, :, :])

        # KV cache management
        if use_cache:
            # Initialize cache if needed
            if self.cache_k is None:
                # Pre-allocate full cache based on max_seq_len to avoid memory fragmentation
                cache_shape = (batch_size, self.n_kv_heads, self.rotary_emb.max_seq_len, self.head_dim)
                self.cache_k = torch.zeros(cache_shape, dtype=k.dtype, device=k.device)
                self.cache_v = torch.zeros(cache_shape, dtype=v.dtype, device=v.device)

            # Update cache with new values
            self.cache_k[:, :, start_pos:start_pos + seq_len, :] = k
            self.cache_v[:, :, start_pos:start_pos + seq_len, :] = v

            # Use cached K, V up to current position
            k = self.cache_k[:, :, :start_pos + seq_len, :]
            v = self.cache_v[:, :, :start_pos + seq_len, :]

        # Repeat K and V for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Attention (with optional FlashAttention)
        if self.use_flash_attn and hasattr(F, 'scaled_dot_product_attention') and not use_cache:
            # Use PyTorch's built-in FlashAttention (SDPA) for prefill only
            # Note: SDPA expects (batch, n_heads, seq_len, head_dim)
            # is_causal=True applies causal mask during attention
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=(mask is not None)
            )
        else:
            # Standard attention (used during decode phase with cache or when FlashAttention disabled)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn + mask
            attn = F.softmax(attn, dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)

    def clear_cache(self):
        """Clear KV cache to free memory"""
        if self.cache_k is not None:
            del self.cache_k
            self.cache_k = None
        if self.cache_v is not None:
            del self.cache_v
            self.cache_v = None


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization"""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, hidden_dim: int, max_seq_len: int = 2048, use_flash_attn: bool = True):
        super().__init__()
        self.attention = Attention(dim, n_heads, n_kv_heads, max_seq_len, use_flash_attn)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), mask, start_pos, use_cache)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Llama3Customized:
    """Pure PyTorch implementation of Llama3 inference"""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_seq_len: int = 2048,
        use_flash_attn: bool = True,
        use_compile: bool = True,
        compile_mode: str = "max-autotune"
    ):
        """
        Initialize Llama3 model

        Args:
            model_name: HuggingFace model name to load weights from
            device: 'cuda' or 'cpu'
            dtype: torch.float16 or torch.float32
            max_seq_len: Maximum sequence length
            use_flash_attn: Whether to use FlashAttention (SDPA)
            use_compile: Whether to use torch.compile for optimization
            compile_mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.use_flash_attn = use_flash_attn
        self.use_compile = use_compile
        self.compile_mode = compile_mode

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load config only first to get model dimensions
        print(f"⏳ Loading model config from {model_name}...")
        config = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).config
        vocab_size = config.vocab_size
        dim = config.hidden_size
        n_layers = config.num_hidden_layers
        n_heads = config.num_attention_heads
        n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
        hidden_dim = config.intermediate_size

        self.vocab_size = vocab_size

        # Build custom model components with correct dimensions
        self.embedding = nn.Embedding(vocab_size, dim).to(device).to(dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, hidden_dim, max_seq_len, use_flash_attn).to(device).to(dtype)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim).to(device).to(dtype)
        self.output = nn.Linear(dim, vocab_size, bias=False).to(device).to(dtype)

        # Tie weights
        self.output.weight = self.embedding.weight

        self.model_components = [self.embedding, *self.layers, self.norm, self.output]

        # Load weights directly from HuggingFace model without keeping full model in memory
        print(f"⏳ Loading weights from {model_name}...")
        self._load_weights_directly(model_name, dtype)

        # Apply PyTorch optimizations
        self._apply_optimizations()

        # Apply torch.compile if requested
        if self.use_compile:
            print(f"⏳ Compiling model with mode '{self.compile_mode}'...")
            self.forward_compiled = torch.compile(
                self.forward,
                mode=self.compile_mode,
                fullgraph=False,
                dynamic=True
            )
            print("✅ Model compilation complete!")
        else:
            self.forward_compiled = None

        # Store reference to self as 'model' for compatibility
        self.model = self

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing = False

        # Print optimization status
        flash_status = "✅ enabled" if self.use_flash_attn and hasattr(F, 'scaled_dot_product_attention') else "❌ disabled"
        compile_status = f"✅ enabled ({self.compile_mode})" if self.use_compile else "❌ disabled"
        print(f"✅ Model loaded successfully! (FlashAttention: {flash_status}, torch.compile: {compile_status})")

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to reduce memory usage during training"""
        import torch.utils.checkpoint as checkpoint
        self.gradient_checkpointing = True
        print("✅ Gradient checkpointing enabled")

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False
        print("✅ Gradient checkpointing disabled")

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        """
        Forward pass

        Args:
            tokens: Input token IDs [batch_size, seq_len]
            start_pos: Starting position for RoPE (for KV cache)
            use_cache: Whether to use KV cache

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = tokens.shape

        # Create causal mask
        # Only needed when seq_len > 1 (prefill phase) or when not using cache
        if seq_len > 1:
            # Prefill: need causal mask
            mask = torch.full((seq_len, seq_len), float("-inf"), device=self.device, dtype=self.dtype)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        else:
            # Decode: single token, no mask needed
            mask = None

        # Forward
        x = self.embedding(tokens)

        # Process each layer with optional gradient checkpointing
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to reduce memory usage during training
                x = torch.utils.checkpoint.checkpoint(
                    self._layer_forward, layer, x, mask, start_pos, use_cache, use_reentrant=False
                )
            else:
                x = layer(x, mask, start_pos, use_cache)

        x = self.norm(x)
        logits = self.output(x)

        return logits

    def _layer_forward(self, layer, x, mask, start_pos, use_cache):
        """Helper function for gradient checkpointing"""
        return layer(x, mask, start_pos, use_cache)

    def clear_cache(self):
        """Clear KV cache for all layers"""
        for layer in self.layers:
            layer.attention.clear_cache()

    def _apply_optimizations(self):
        """Apply PyTorch performance optimizations"""
        if self.device == "cuda":
            # Enable TF32 for matmul operations (Ampere GPUs and newer)
            torch.backends.cuda.matmul.allow_tf32 = True
            # Enable TF32 for cuDNN operations
            torch.backends.cudnn.allow_tf32 = True
            # Enable cuDNN benchmarking for optimal kernels
            torch.backends.cudnn.benchmark = True
            # Set matmul precision for faster computation
            torch.set_float32_matmul_precision('high')

            print("✅ Applied CUDA optimizations (TF32, cuDNN benchmark)")
    
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,  # Change to be compatible with both tensor and string
        prompt: str = None,
        attention_mask=None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = 50,
        do_sample: bool = True,
        pad_token_id=None,
        eos_token_id=None,
        use_cache: bool = True,
        max_memory_gb: Optional[float] = None,  # New parameter: maximum memory usage in GB
        **kwargs
    ):
        """
        Generate text from prompt or input_ids using custom forward pass
        Compatible with HuggingFace generate API

        Args:
            input_ids: Input token IDs [batch_size, seq_len] (alternative to prompt)
            prompt: Input text (alternative to input_ids)
            attention_mask: Attention mask (for padding, currently ignored in custom implementation)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (optional)
            do_sample: Whether to sample (if False, uses greedy decoding)
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            use_cache: Whether to use KV cache for faster generation
            max_memory_gb: Maximum allowed GPU memory usage in GB (stops generation if exceeded)
            **kwargs: Additional arguments (ignored)

        Returns:
            If input_ids provided: Generated token IDs [batch_size, seq_len + max_new_tokens]
            If prompt provided: Generated text string
        """
        # Handle input - check if input_ids is actually a string (first positional arg is prompt)
        if input_ids is not None:
            # Check if input_ids is a string, which means it's actually the prompt
            if isinstance(input_ids, str):
                prompt = input_ids
                input_ids = None

        # Handle input
        if input_ids is not None:
            tokens = input_ids.to(self.device)
            return_text = False
        elif prompt is not None:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            tokens = inputs.input_ids.to(self.device)
            return_text = True
        else:
            raise ValueError("Either input_ids or prompt must be provided")

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        batch_size = tokens.shape[0]

        # Clear cache before generation
        if use_cache:
            self.clear_cache()

        # Track current position for RoPE
        cur_pos = 0

        # Generate tokens autoregressively
        for i in range(max_new_tokens):
            # Check memory usage and exit early if needed
            if max_memory_gb is not None and self.device == "cuda":
                current_memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                if current_memory_gb > max_memory_gb:
                    print(f"⚠️ Memory limit exceeded ({current_memory_gb:.2f}GB > {max_memory_gb}GB). Stopping generation.")
                    break

            # Use compiled forward if available
            forward_fn = self.forward_compiled if self.forward_compiled is not None else self.forward

            # With KV cache: process full sequence on first pass, then only last token
            if use_cache:
                if i == 0:
                    # Prefill: process full sequence and initialize cache
                    logits = forward_fn(tokens, start_pos=0, use_cache=True)[:, -1, :]
                    cur_pos = tokens.shape[1]
                else:
                    # Decode: only process the last token using cache
                    current_tokens = tokens[:, -1:]
                    logits = forward_fn(current_tokens, start_pos=cur_pos, use_cache=True)[:, -1, :]
                    cur_pos += 1
            else:
                # No cache: process full sequence every time
                logits = forward_fn(tokens, start_pos=0, use_cache=False)[:, -1, :]

            if do_sample and temperature > 0:
                # Apply temperature
                logits = logits / temperature

                # Top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[..., 0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Check max length
            if tokens.shape[1] >= self.max_seq_len:
                break

        # Clear cache after generation
        if use_cache:
            self.clear_cache()

        # Return based on input type
        if return_text:
            return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        else:
            return tokens
    
    def _load_weights_directly(self, model_name: str, dtype: torch.dtype):
        """Load weights directly from HuggingFace model without keeping full model in memory"""
        import transformers
        from transformers.utils import cached_file, SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
        from transformers.modeling_utils import load_state_dict
        import os
        import json
        try:
            from safetensors.torch import load_file
        except ImportError:
            # If safetensors is not available, fall back to pytorch format
            model_path = cached_file(model_name, transformers.modeling_utils.WEIGHTS_NAME)
            state_dict = load_state_dict(model_path)
            load_file = None

        # First try to find safetensors files if the module is available
        state_dict = None
        if load_file is not None:
            try:
                # Try loading the index file first (sharded safetensors)
                index_file = cached_file(model_name, SAFE_WEIGHTS_INDEX_NAME)
                with open(index_file, 'r') as f:
                    index = json.load(f)

                # Load all safetensors files
                state_dict = {}
                for safetensors_file in set(index['weight_map'].values()):
                    safetensors_path = cached_file(model_name, safetensors_file)
                    partial_state_dict = load_file(safetensors_path)
                    state_dict.update(partial_state_dict)

            except:  # If index file doesn't exist or fails, try single safetensors file
                try:
                    model_path = cached_file(model_name, SAFE_WEIGHTS_NAME)
                    state_dict = load_file(model_path)
                except:
                    # Fall back to pytorch format
                    model_path = cached_file(model_name, transformers.modeling_utils.WEIGHTS_NAME)
                    state_dict = load_state_dict(model_path)
        else:
            # safetensors not available, use pytorch format
            model_path = cached_file(model_name, transformers.modeling_utils.WEIGHTS_NAME)
            state_dict = load_state_dict(model_path)

        # Load embedding weights
        if 'model.embed_tokens.weight' in state_dict:
            self.embedding.weight.data.copy_(state_dict['model.embed_tokens.weight'].to(self.device).to(dtype))

        # Load layer weights
        for i, layer in enumerate(self.layers):
            prefix = f'model.layers.{i}.'

            # Attention weights
            if f'{prefix}self_attn.q_proj.weight' in state_dict:
                layer.attention.q_proj.weight.data.copy_(state_dict[f'{prefix}self_attn.q_proj.weight'].to(self.device).to(dtype))
            if f'{prefix}self_attn.k_proj.weight' in state_dict:
                layer.attention.k_proj.weight.data.copy_(state_dict[f'{prefix}self_attn.k_proj.weight'].to(self.device).to(dtype))
            if f'{prefix}self_attn.v_proj.weight' in state_dict:
                layer.attention.v_proj.weight.data.copy_(state_dict[f'{prefix}self_attn.v_proj.weight'].to(self.device).to(dtype))
            if f'{prefix}self_attn.o_proj.weight' in state_dict:
                layer.attention.o_proj.weight.data.copy_(state_dict[f'{prefix}self_attn.o_proj.weight'].to(self.device).to(dtype))

            # Feed-forward weights
            if f'{prefix}mlp.gate_proj.weight' in state_dict:
                layer.feed_forward.gate_proj.weight.data.copy_(state_dict[f'{prefix}mlp.gate_proj.weight'].to(self.device).to(dtype))
            if f'{prefix}mlp.up_proj.weight' in state_dict:
                layer.feed_forward.up_proj.weight.data.copy_(state_dict[f'{prefix}mlp.up_proj.weight'].to(self.device).to(dtype))
            if f'{prefix}mlp.down_proj.weight' in state_dict:
                layer.feed_forward.down_proj.weight.data.copy_(state_dict[f'{prefix}mlp.down_proj.weight'].to(self.device).to(dtype))

            # Layer norms
            if f'{prefix}input_layernorm.weight' in state_dict:
                layer.attention_norm.weight.data.copy_(state_dict[f'{prefix}input_layernorm.weight'].to(self.device).to(dtype))
            if f'{prefix}post_attention_layernorm.weight' in state_dict:
                layer.ffn_norm.weight.data.copy_(state_dict[f'{prefix}post_attention_layernorm.weight'].to(self.device).to(dtype))

        # Load final norm
        if 'model.norm.weight' in state_dict:
            self.norm.weight.data.copy_(state_dict['model.norm.weight'].to(self.device).to(dtype))

        # Load output projection (lm_head)
        if 'lm_head.weight' in state_dict:
            self.output.weight.data.copy_(state_dict['lm_head.weight'].to(self.device).to(dtype))

        # Clean up the loaded state dict to free memory
        del state_dict
        torch.cuda.empty_cache()

    def _load_from_hf_model(self, hf_model):
        """Load weights from HuggingFace model to custom architecture"""
        hf_state = hf_model.state_dict()

        # Load embedding weights
        if 'model.embed_tokens.weight' in hf_state:
            self.embedding.weight.data.copy_(hf_state['model.embed_tokens.weight'])

        # Load layer weights
        for i, layer in enumerate(self.layers):
            prefix = f'model.layers.{i}.'

            # Attention weights
            if f'{prefix}self_attn.q_proj.weight' in hf_state:
                layer.attention.q_proj.weight.data.copy_(hf_state[f'{prefix}self_attn.q_proj.weight'])
            if f'{prefix}self_attn.k_proj.weight' in hf_state:
                layer.attention.k_proj.weight.data.copy_(hf_state[f'{prefix}self_attn.k_proj.weight'])
            if f'{prefix}self_attn.v_proj.weight' in hf_state:
                layer.attention.v_proj.weight.data.copy_(hf_state[f'{prefix}self_attn.v_proj.weight'])
            if f'{prefix}self_attn.o_proj.weight' in hf_state:
                layer.attention.o_proj.weight.data.copy_(hf_state[f'{prefix}self_attn.o_proj.weight'])

            # Feed-forward weights
            if f'{prefix}mlp.gate_proj.weight' in hf_state:
                layer.feed_forward.gate_proj.weight.data.copy_(hf_state[f'{prefix}mlp.gate_proj.weight'])
            if f'{prefix}mlp.up_proj.weight' in hf_state:
                layer.feed_forward.up_proj.weight.data.copy_(hf_state[f'{prefix}mlp.up_proj.weight'])
            if f'{prefix}mlp.down_proj.weight' in hf_state:
                layer.feed_forward.down_proj.weight.data.copy_(hf_state[f'{prefix}mlp.down_proj.weight'])

            # Layer norms
            if f'{prefix}input_layernorm.weight' in hf_state:
                layer.attention_norm.weight.data.copy_(hf_state[f'{prefix}input_layernorm.weight'])
            if f'{prefix}post_attention_layernorm.weight' in hf_state:
                layer.ffn_norm.weight.data.copy_(hf_state[f'{prefix}post_attention_layernorm.weight'])

        # Load final norm
        if 'model.norm.weight' in hf_state:
            self.norm.weight.data.copy_(hf_state['model.norm.weight'])

        # Load output projection (lm_head)
        if 'lm_head.weight' in hf_state:
            self.output.weight.data.copy_(hf_state['lm_head.weight'])

    def load_weights(self, state_dict: dict):
        """Load model weights from state dictionary"""
        # Map state dict to model components
        self.embedding.load_state_dict(
            {k.replace('tok_embeddings.', ''): v for k, v in state_dict.items()
             if k.startswith('tok_embeddings')}
        )

        for i, layer in enumerate(self.layers):
            layer_prefix = f'layers.{i}.'
            layer_dict = {k.replace(layer_prefix, ''): v
                         for k, v in state_dict.items() if k.startswith(layer_prefix)}
            layer.load_state_dict(layer_dict, strict=False)

        self.norm.load_state_dict(
            {k.replace('norm.', ''): v for k, v in state_dict.items()
             if k.startswith('norm.')}
        )

        self.output.load_state_dict(
            {k.replace('output.', ''): v for k, v in state_dict.items()
             if k.startswith('output.')}
        )
    
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return 0.0
    
    def get_model_size(self) -> float:
        """Get model parameter size in GB"""
        param_size = sum(
            p.numel() * p.element_size()
            for component in self.model_components
            for p in component.parameters()
        )
        return param_size / (1024 ** 3)