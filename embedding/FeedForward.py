import numpy as np
import numpy.typing as npt
from typing import cast


Array = npt.NDArray[np.float64]


class FeedForward:
    """Position-wise feed-forward network used in Transformer blocks.

    The module applies two learned linear projections with a ReLU non-linearity:
    ``x -> ReLU(x @ W1 + b1) @ W2 + b2``.

    Args:
        embedding_dim: Input and output feature size.
        hidden_dim: Intermediate projection size.

    Attributes:
        W1: First projection matrix of shape ``(embedding_dim, hidden_dim)``.
        b1: Bias for first projection of shape ``(hidden_dim,)``.
        W2: Second projection matrix of shape ``(hidden_dim, embedding_dim)``.
        b2: Bias for second projection of shape ``(embedding_dim,)``.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        self.embedding_dim: int = embedding_dim
        self.hidden_dim: int = hidden_dim
        self.W1: Array = cast(Array, np.random.randn(embedding_dim, hidden_dim) * 0.01)
        self.b1: Array = np.zeros(hidden_dim)
        self.W2: Array = cast(Array, np.random.randn(hidden_dim, embedding_dim) * 0.01)
        self.b2: Array = np.zeros(embedding_dim)

    @staticmethod
    def relu(x: Array) -> Array:
        """Apply ReLU activation element-wise."""
        return np.maximum(0, x)

    def forward(self, x: Array) -> Array:
        """Run a forward pass for an input tensor of shape ``(..., embedding_dim)``."""
        hidden = self.relu(cast(Array, x @ self.W1 + self.b1))
        output = cast(Array, hidden @ self.W2 + self.b2)
        return output


class FeeedForward(FeedForward):
    """Backward-compatible alias for the misspelled class name `FeeedForward`."""


def run() -> None:
    """Run a small FeedForward demo from the command line."""
    np.random.seed(42)
    embedding_dim = 8
    hidden_dim = 16
    seq_length = 5

    ff = FeedForward(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    x: Array = cast(Array, np.random.randn(seq_length, embedding_dim))
    y = ff.forward(x)

    print("FeedForward demo")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")


if __name__ == "__main__":
    run()








