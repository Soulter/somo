from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import my_functional as MyF


@dataclass
class GPTConfig:
    vocab_size: int  # how many tokens in our tokenizer
    seq_len: int  # context window
    n_layers: int  # how many layers in our transformer block
    n_heads: int  # heads number
    d_model: int  # how many dimentions a token is presented
    dropout: float = 0.0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x_norm


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = (
            config.d_model // config.n_heads
        )  # the number of dimentions that allocate to a head
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        """
        Same as:
            self.q_proj = nn.Linear(C, C)
            self.k_proj = nn.Linear(C, C)
            self.v_proj = nn.Linear(C, C)
        """
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq length, hidden size / d_model
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # multiple heads split
        # transpose: exchange the position of self.n_heads and T
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # apply dot product attention
        # apply this attention magic for every head.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
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
        # position embedding
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)  # T, C

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.norm = RMSNorm(config.d_model)
        # the last output layer
        # [d_model] -> [vocab_size]
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,  # B, T
        targets: torch.Tensor | None = None,  # B, T
    ):
        B, T = idx.shape
        assert T <= self.config.seq_len, (
            "Cannot forward, model block size is exhausted."
        )

        pos = torch.arange(0, T, device=idx.device)
        tok = self.token_emb(idx)  # b,t,c
        pos = self.pos_emb(pos)  # t,c
        x = tok + pos  # broadcast

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
