import torch


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
):
    # q, k, v: [B, n_heads, T, head_dim]
    # which means:
    # number of samples in a batch, heads number in a token, tokens number in a batch, heads dimension in a head.
    head_dim = q.size(-1)

    scale = head_dim**-0.5
    scores = q @ k.transpose(-2, -1) * scale
    # generate mask
    if attn_mask is None and is_causal:
        T = v.size(-2)
        # triu is upper triangular matrix, which means the elements below the diagonal are all 0, and the elements above the diagonal are all 1.
        # like this:
        # [[0, 1, 1, 1],
        #  [0, 0, 1, 1],
        #  [0, 0, 0, 1],
        #  [0, 0, 0, 0]]
        attn_mask = torch.triu(
            torch.ones(T, T, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
    scores = scores.masked_fill(attn_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nn.functional.dropout(attn, p=dropout_p)
    output = attn @ v
    return output
