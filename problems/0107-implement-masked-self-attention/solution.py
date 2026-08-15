import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    d_k = K.shape[-1]
    logits = Q @ K.transpose(-2, -1) / d_k ** 0.5

    if mask is not None:
        logits += mask
    
    logits = torch.softmax(logits, dim=-1)

    return logits @ V