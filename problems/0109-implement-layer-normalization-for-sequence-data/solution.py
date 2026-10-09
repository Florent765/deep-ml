import torch

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization.
    """
    var, mean = torch.var_mean(X, dim=-1, correction=0, keepdim=True)
    out = ((X - mean) / (torch.sqrt(var + epsilon))) * gamma + beta
    return out