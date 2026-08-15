import torch
import torch.nn.functional as F

def grouped_query_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """
    Compute Grouped Query Attention.

    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads

    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    head_dim = Q.shape[-1] // num_heads
    num_queries_per_kv = num_heads // num_kv_heads
    Q = Q.unflatten(-1, (num_heads, -1)).transpose(-3, -2)
    K = K.unflatten(-1, (num_kv_heads, -1)).transpose(-3, -2)
    V = V.unflatten(-1, (num_kv_heads, -1)).transpose(-3, -2)


    K = K.unsqueeze(dim=-3).expand(-1, -1, num_queries_per_kv, -1, -1).flatten(-4, -3)
    V = V.unsqueeze(dim=-3).expand(-1, -1, num_queries_per_kv, -1, -1).flatten(-4, -3)

    logits = Q @ K.transpose(-2, -1) / head_dim ** 0.5
    logits = torch.softmax(logits, dim=-1)

    logits = logits @ V
    logits = logits.transpose(-3, -2).flatten(-2)

    return logits