import torch

def multiquery_attention(X: torch.Tensor, W_queries: list, W_key: torch.Tensor, W_value: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_queries: List of query weight tensors, each (d_model, d_k), one per head
        W_key: Shared key weight tensor of shape (d_model, d_k)
        W_value: Shared value weight tensor of shape (d_model, d_v)
        W_out: Output projection tensor of shape (num_heads * d_v, d_model)
    
    Returns:
        Output tensor of shape (seq_len, d_model), rounded to 4 decimal places
    """
    K = X @ W_key # (T, d_k)
    V = X @ W_value # (T, d_v)
    W_Q = torch.stack(W_queries)
    Q = X[None, :] @ W_Q # (H, T, d_k)

    scale_factor = 1 / K.shape[-1]**0.5

    logits = Q @ K[None, :].transpose(-2, -1) * scale_factor # (H, T, T)

    max_val = torch.max(logits, dim=-1, keepdim=True).values # (H, T, 1)
    logits_shifted = logits - max_val # (H, T, T)

    exp = torch.exp(logits_shifted)
    sum_exp = torch.sum(exp, dim=-1, keepdim=True)
    logits = exp / sum_exp # (H, T, T)

    attn = logits @ V[None, :] # (H, T, d_v)
    attn = attn.transpose(-3, -2).flatten(start_dim=-2) # (T, H*d_v)

    return attn @ W_out