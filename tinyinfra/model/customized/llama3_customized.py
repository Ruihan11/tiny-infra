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
    # Take only the first half to match x1/x2 dimensions
    d = x1.shape[-1]
    return torch.cat([
        x1 * cos[..., :d] - x2 * sin[..., :d],
        x2 * cos[..., :d] + x1 * sin[..., :d]
    ], dim=-1)


class Attention(nn.Module):
    """Multi-head attention with RoPE and Grouped Query Attention (GQA)"""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = None, max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(x, seq_len)
        q = apply_rotary_emb(q, cos[None, None, :, :], sin[None, None, :, :])
        k = apply_rotary_emb(k, cos[None, None, :, :], sin[None, None, :, :])

        # Repeat K and V for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


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
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, hidden_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.attention = Attention(dim, n_heads, n_kv_heads, max_seq_len)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Llama3Customized:
    """Pure PyTorch implementation of Llama3 inference"""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_seq_len: int = 2048
    ):
        """
        Initialize Llama3 model

        Args:
            model_name: HuggingFace model name to load weights from
            device: 'cuda' or 'cpu'
            dtype: torch.float16 or torch.float32
            max_seq_len: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load pretrained model to get config
        print(f"⏳ Loading pretrained weights from {model_name}...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # Get model config from HuggingFace model
        config = hf_model.config
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
            TransformerBlock(dim, n_heads, n_kv_heads, hidden_dim, max_seq_len).to(device).to(dtype)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim).to(device).to(dtype)
        self.output = nn.Linear(dim, vocab_size, bias=False).to(device).to(dtype)

        # Tie weights
        self.output.weight = self.embedding.weight

        self.model_components = [self.embedding, *self.layers, self.norm, self.output]

        # Load weights from HuggingFace model
        self._load_from_hf_model(hf_model)
        del hf_model  # Free memory
        torch.cuda.empty_cache()

        # Store reference to self as 'model' for compatibility
        self.model = self
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            tokens: Input token IDs [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = tokens.shape
        
        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        mask = mask.view(1, 1, seq_len, seq_len)
        
        # Forward
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        logits = self.output(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor = None,
        prompt: str = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
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
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)
            do_sample: Whether to sample (if False, uses greedy decoding)
            pad_token_id: Padding token ID
            **kwargs: Additional arguments (ignored)

        Returns:
            If input_ids provided: Generated token IDs [batch_size, seq_len + max_new_tokens]
            If prompt provided: Generated text string
        """
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

        batch_size = tokens.shape[0]

        # Generate tokens autoregressively using custom forward
        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = self.forward(tokens)[:, -1, :]

            if do_sample and temperature > 0:
                # Apply temperature
                logits = logits / temperature

                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

            # Check max length
            if tokens.shape[1] >= self.max_seq_len:
                break

        # Return based on input type
        if return_text:
            return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        else:
            return tokens
    
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