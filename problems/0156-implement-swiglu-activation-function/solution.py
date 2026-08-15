import torch

def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor of shape (batch_size, 2d)
    Returns:
        torch.Tensor of shape (batch_size, d)
    """
    x_val, x_gate = torch.chunk(x, 2, dim=-1)

    return x_val * (x_gate * torch.sigmoid(x_gate))