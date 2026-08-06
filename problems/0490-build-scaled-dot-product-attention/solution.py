import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	d_k = K.shape[-1]
	scale = 1 / np.sqrt(d_k)
	
	logits = (Q @ np.swapaxes(K, -1, -2)) * scale

	if mask is not None:
		logits = np.where(mask == 0, -1e9, logits)

	max_value = np.max(logits, axis=-1, keepdims=True)
	num = np.exp(logits - max_value)
	den = np.sum(num, axis=-1, keepdims=True)

	softmax = num / den

	return softmax @ V, softmax
