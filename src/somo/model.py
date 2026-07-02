from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int  # how many tokens in our tokenizer
    seq_len: int  # context window
    n_layers: int  # how many layers in our transformer block
    n_heads: int  # heads number
    d_model: int  # how many dimentions a token is presented
    dropout: float = 0.0
    n_kv_heads: int | None = None # if None, then n_kv_heads = n_heads

    qk_norm: bool = False # whether to normalize q and k before applying RoPE

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_heads % self.n_kv_heads == 0


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x_norm


def build_rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0):
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: [B, H, T, D]. RoPE rotates each even/odd pair in the head dimension.
    T = x.size(-2)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = cos[:, :, :T, :].to(dtype=x.dtype)
    sin = sin[:, :, :T, :].to(dtype=x.dtype)

    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        # the number of dimentions that allocate to a head
        self.head_dim = config.d_model // config.n_heads
        self.n_kv_heads = config.n_kv_heads
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # MHA
        # self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        """
        MHA Same as:
            self.q_proj = nn.Linear(C, C)
            self.k_proj = nn.Linear(C, C)
            self.v_proj = nn.Linear(C, C)
        """

        # GQA
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        rope_cos, rope_sin = build_rope_cache(config.seq_len, self.head_dim)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq length, hidden size / d_model
        # qkv = self.qkv_proj(x)
        # q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # multiple heads split
        # transpose: exchange the position of self.n_heads and T
        # q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # normalize q and k before applying RoPE
        if hasattr(self, "q_norm") and hasattr(self, "k_norm"):
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, self.rope_cos, self.rope_sin) # type: ignore
        k = apply_rope(k, self.rope_cos, self.rope_sin) # type: ignore

        # apply dot product attention
        # apply this attention magic for every head.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=self.n_kv_heads != self.n_heads,
        )

        #
        # [B, n_heads, T, head_dim]
        #    -> [B, T, n_heads, head_dim]
        #    -> [B, T, C]
        #
        # contiguous: to make memory contiguous after using transpose / permute.
        # pytorch use stride to record data after transposing

        # or we can use .reshape(B, T, C) to replace .contiguous().view(B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.out_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # why 4? because in the original transformer paper,
        # the hidden dimension of the feedforward network is 4 times the model dimension.
        # This is a common practice in transformer architectures to allow for
        # more expressive power in the feedforward layers.
        hidden_dim = 4 * config.d_model
        self.fc1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        # gelu?
        # GELU (Gaussian Error Linear Unit) is an activation function that is used in neural networks,
        # particularly in transformer architectures. It is defined as:
        # GELU(x) = x * Φ(x)
        # where Φ(x) is the cumulative distribution function of the standard normal distribution.
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # input: B, T, C; output: B, T, C
        self.attn_norm = RMSNorm(d_model=config.d_model)
        # input: B, T, C; output: B, T, C
        self.attn = CausalSelfAttention(config=config)

        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = MLP(config=config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))  # residual connection
        x = x + self.mlp(self.mlp_norm(x))  # residual connection
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # learnable vocab
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)  # B, T, C

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.norm = RMSNorm(config.d_model)
        # the last output layer
        # [d_model] -> [vocab_size]
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_emb.weight # weights tying: the same embedding matrix is used for both the input and output embeddings, which can help improve generalization and reduce the number of parameters in the model.

    def _init_weights(self, module):
        # must add this,
        # or the loss will start from 500+,
        # because the initial weights are too large, and the softmax will be very small, leading to a large loss.
        if isinstance(module, nn.Linear):
            # initialization for linear layers
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            # initialization for embedding layers
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,  # B, T
        targets: torch.Tensor | None = None,  # B, T
    ):
        B, T = idx.shape
        assert T <= self.config.seq_len, (
            "Cannot forward, model block size is exhausted."
        )

        x = self.token_emb(idx)  # b,t,c

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # logits[0, 0, 2] represents the score of token id 2 in the token position 0 in the batch 0
        logits = self.lm_head(x)  # b,t,vocab_size.

        # if provides answers -> calculate loss
        loss = None
        if targets is not None:
            # loss = -log( softmax(logits)[target] )
            # softmax: p_i = exp(z_i) / sum_j exp(z_j)

            # -1 mens calculate dimension automatically -> single -1 in a .view() at most!
            # or: RuntimeError: only one dimension can be inferred
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # [B, T, V] -> [B*T, V]
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,  # B, T
        max_new_tokens: int,
        temperature: float = 1.0,
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.seq_len :]  # context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # only need the predict

            # t < 1: more conservative: after softmax, the largest logit dominates more;
            # t > 1: more random:
            #   divide by a larger number, so the differences get smaller.
            #   After softmax, probabilities become more spread out. Lower-ranked tokens get more chance.
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # randomness! [B, 1]
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
