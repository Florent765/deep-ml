import torch
import math

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                           block_size: int = 2) -> torch.Tensor:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (seq_len, d_model)
        K: Key matrix (seq_len, d_model)
        V: Value matrix (seq_len, d_model)
        block_size: Size of blocks for tiled computation
    
    Returns:
        Output matrix (seq_len, d_model)
    """
    seq_len, d_model = Q.shape

	O = torch.zeros(seq_len, d_model)
	L = torch.zeros(seq_len, 1)
	m = torch.full((seq_len, 1), float("-inf"))

	T = math.ceil(seq_len / block_size)

    for i in range(T):
        q_start, q_end = i * block_size, min((i + 1) * block_size, seq_len)

        Q_i = Q[q_start:q_end, :]

        O_i = O[q_start:q_end, :]
        l_i = L[q_start:q_end, :]
        m_i = m[q_start:q_end, :]

        for j in range(T):
            start, end = j * block_size, min((j + 1) * block_size, seq_len)

            K_j = K[start:end, :]
            V_j = V[start:end, :]

            S_ij = Q_i @ K_j.transpose(-2, -1) / d_model ** 0.5
            
            m_ij = torch.max(S_ij, dim=-1, keepdim=True).values
            m_new = torch.maximum(m_i, m_ij)

            P_ij = torch.exp(S_ij - m_new)
            scale = torch.exp(m_i - m_new)

            l_new = scale * l_i + torch.sum(P_ij, dim=-1, keepdim=True)

            O_i = scale * O_i + P_ij @ V_j

            m_i = m_new
            l_i = l_new

        O[q_start:q_end, :] = O_i / l_i
        L[q_start:q_end, :] = l_i
        m[q_start:q_end, :] = m_i
    
    return O