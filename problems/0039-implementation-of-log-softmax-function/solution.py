import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    scores = torch.tensor(scores)
	max_val = torch.max(scores, dim=-1, keepdim=True).values

	exp = torch.exp(scores)
	sum_exp = torch.sum(exp, dim=-1, keepdim=True)

	return scores - torch.log(sum_exp)
