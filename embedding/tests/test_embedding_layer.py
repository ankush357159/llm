import pytest
import numpy as np
from embedding.EmbeddingLayer import EmbeddingLayer


def test_embedding_matrix_shape_and_initialization():
    """Verify the embedding matrix initializes with the correct shape and scale."""
    vocab_size = 10
    embedding_dim = 8

    layer = EmbeddingLayer(vocab_size, embedding_dim)

    # Check dimensions
    assert layer.embedding_matrix.shape == (vocab_size, embedding_dim)

    # Check that initialization values are small numbers scaled by 0.01 (0.0 to 0.01)
    assert np.all(layer.embedding_matrix >= 0.0)
    assert np.all(layer.embedding_matrix <= 0.01)


def test_forward_single_sequence_1d():
    """Verify forward pass returns correct shapes for a single sequence (1D input)."""
    vocab_size = 5
    embedding_dim = 4
    layer = EmbeddingLayer(vocab_size, embedding_dim)

    # 3 token IDs
    token_ids = [0, 2, 4]
    vectors = layer.forward(token_ids)

    # Expected output shape: (sequence_length, embedding_dim) -> (3, 4)
    assert vectors.shape == (3, 4)

    # Verify that row 0 of output exactly matches row 0 of matrix
    assert np.array_equal(vectors[0], layer.embedding_matrix[0])
    # Verify that row 1 of output exactly matches row 2 of matrix
    assert np.array_equal(vectors[1], layer.embedding_matrix[2])


def test_forward_batch_sequences_2d():
    """Verify forward pass works perfectly with a 2D batch input."""
    vocab_size = 6
    embedding_dim = 3
    layer = EmbeddingLayer(vocab_size, embedding_dim)

    # A batch of 2 sentences, each with 3 tokens (Shape: 2 x 3)
    token_ids_batch = [
        [1, 3, 5],
        [0, 2, 4]
    ]

    vectors = layer.forward(token_ids_batch)

    # Expected output shape: (batch_size, sequence_length, embedding_dim) -> (2, 3, 3)
    assert vectors.shape == (2, 3, 3)


def test_invalid_index_raises_error():
    """Verify that passing a token ID outside the vocabulary boundary crashes safely."""
    vocab_size = 5
    embedding_dim = 4
    layer = EmbeddingLayer(vocab_size, embedding_dim)

    # Token ID 5 is out of bounds for a vocab_size of 5 (valid IDs are 0 to 4)
    invalid_token_ids = [1, 5]

    with pytest.raises(IndexError):
        layer.forward(invalid_token_ids)
