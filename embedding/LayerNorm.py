import numpy as np
class LayerNorm:
    """
    Applies Layer Normalization over the last dimension of the input array
    to stabilize neural network hidden states by forcing a mean of 0
    and variance of 1, scaled and shifted by learnable weights.
    """
    def __init__(self, embedding_dim, eps=1e-6):
        self.gamma = np.ones(embedding_dim)
        self.beta = np.zeros(embedding_dim)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta