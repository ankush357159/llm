from typing import cast

import numpy as np
import numpy.typing as npt

from embedding.MultiHeadAttention import MultiHeadAttention


Array = npt.NDArray[np.float64]


def test_multi_head_attention():
    """Verifies integration, dimensions, and shape logic for Multi-Head Attention."""
    # Setup matrix parameters
    embedding_dim = 64
    num_heads = 4
    seq_length = 12

    # 1. Initialize the class
    mha = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads)

    # 2. Assert initialization settings are correct
    assert mha.head_dim == 16, "Head dimension calculation failed."
    assert len(mha.heads) == 4, "Failed to initialize 4 attention heads."

    # 3. Create dummy input data matrix (sequence_length, embedding_dim)
    dummy_input = np.random.randn(seq_length, embedding_dim)

    # 4. Run the forward pipeline
    output = cast(Array, mha.forward(dummy_input))

    # 5. Assert final output shape integrity
    expected_shape = (seq_length, embedding_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
