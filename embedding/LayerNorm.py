import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.float64]


class LayerNorm:
    """
    Applies Layer Normalization over the last dimension of the input array
    to stabilize neural network hidden states by forcing a mean of 0
    and variance of 1, scaled and shifted by learnable weights.
    """
    def __init__(self, embedding_dim: int, eps: float = 1e-6):
        self.gamma: Array = np.ones(embedding_dim)
        self.beta: Array = np.zeros(embedding_dim)
        self.eps: float = eps

    def forward(self, x: Array) -> Array:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta