import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    scores = torch.tensor(scores)
    max_val = torch.max(scores, dim=-1, keepdim=True).values

    exp = torch.exp(scores - max_val)
    sum_exp = torch.sum(exp, dim=-1, keepdim=True)

    return (exp / sum_exp).tolist()
