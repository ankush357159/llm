from typing import cast

import numpy as np
import numpy.typing as npt

from embedding.SelfAttention import SelfAttention


Array = npt.NDArray[np.float64]


class MultiHeadAttention:
    """Multi-Head Attention mechanism using a list of single-head attention blocks.

    This class splits the input embedding space into multiple smaller heads,
    applies self-attention to each head independently, concatenates the
    results, and projects them back to the original embedding dimension.

    Parameters
    ----------
    embedding_dim : int
        The total dimension of the input feature vector.
    num_heads : int
        The number of parallel attention heads. Must divide `embedding_dim` evenly.

    Attributes
    ----------
    embedding_dim : int
        Total feature dimension of the input.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Feature dimension allocated per head (`embedding_dim // num_heads`).
    heads : list of SelfAttention
        List containing the individual single-head attention modules.
    W_o : numpy.ndarray
        Output projection weight matrix of shape `(embedding_dim, embedding_dim)`.

    Raises
    ------
    AssertionError
        If `embedding_dim` is not perfectly divisible by `num_heads`.
    """

    def __init__(self, embedding_dim: int, num_heads: int):
        assert embedding_dim % num_heads == 0

        self.embedding_dim: int = embedding_dim
        self.num_heads: int = num_heads
        self.head_dim: int = embedding_dim // num_heads

        self.heads: list[SelfAttention] = [SelfAttention(self.head_dim) for _ in range(num_heads)]
        self.W_o: Array = cast(Array, np.random.randn(embedding_dim, embedding_dim) * 0.01)

    def split_heads(self, x: Array) -> Array:
        """Splits the input sequence into multiple heads.

        Parameters
        ----------
        x : numpy.ndarray
            Input array of shape `(seq_length, embedding_dim)`.

        Returns
        -------
        numpy.ndarray
            Reshaped array of shape `(seq_length, num_heads, head_dim)`.
        """
        seq_length = x.shape[0]
        return x.reshape(seq_length, self.num_heads, self.head_dim)

    def forward(self, x: Array) -> Array:
        """Executes the multi-head attention forward pass.

        Parameters
        ----------
        x : numpy.ndarray
            Input sequence matrix of shape `(seq_length, embedding_dim)`.

        Returns
        -------
        numpy.ndarray
            Projected attention output of shape `(seq_length, embedding_dim)`.
        """
        split_x = self.split_heads(x)
        head_outputs: list[Array] = []
        for i in range(self.num_heads):
            head_input = split_x[:, i, :]
            head_output = cast(Array, self.heads[i].forward(head_input))
            head_outputs.append(head_output)

        concatenated = np.concatenate(head_outputs, axis=-1)
        output = cast(Array, concatenated @ self.W_o)
        return output


def run() -> None:
    """Run a minimal demonstration of MultiHeadAttention."""
    np.random.seed(42)

    embedding_dim = 64
    num_heads = 4
    seq_length = 12

    mha = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads)
    dummy_input: Array = cast(Array, np.random.randn(seq_length, embedding_dim))
    output = mha.forward(dummy_input)

    print("MultiHeadAttention demo")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    run()


