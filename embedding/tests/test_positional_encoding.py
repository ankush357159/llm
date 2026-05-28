import pytest
import numpy as np
from embedding.PositionalEncoding import PositionalEncoding


def test_encoding_matrix_shape():
    """Verify the internal encoding matrix initializes with correct max dimensions."""
    max_len = 50
    embedding_dim = 16
    pe = PositionalEncoding(max_len, embedding_dim)

    assert pe.encoding.shape == (max_len, embedding_dim)


def test_forward_single_sentence_2d():
    """Verify forward pass correctly processes a single 2D sentence matrix (seq_len, dim)."""
    max_len = 10
    embedding_dim = 4
    pe = PositionalEncoding(max_len, embedding_dim)

    # Simulate a single sentence with 3 tokens, each with 4 dimensions
    seq_len = 3
    x_single = np.ones((seq_len, embedding_dim))

    output = pe.forward(x_single)

    # Shape must remain exactly the same
    assert output.shape == (seq_len, embedding_dim)

    # Verify values: since input was all 1s, output must equal (1 + positional_values)
    expected_output = 1.0 + pe.encoding[:seq_len]
    assert np.allclose(output, expected_output)


def test_forward_batch_sentences_3d():
    """Verify forward pass handles a 3D batch tensor (batch_size, seq_len, dim)."""
    max_len = 10
    embedding_dim = 4
    pe = PositionalEncoding(max_len, embedding_dim)

    # Simulate a batch of 2 sentences, 3 tokens each, 4 dimensions
    batch_size = 2
    seq_len = 3
    x_batch = np.ones((batch_size, seq_len, embedding_dim))

    output = pe.forward(x_batch)

    # Shape must match the 3D batch shape perfectly
    assert output.shape == (batch_size, seq_len, embedding_dim)

    # Verify that broadcasting worked across both sentences in the batch
    expected_sentence_values = 1.0 + pe.encoding[:seq_len]
    assert np.allclose(output[0], expected_sentence_values)
    assert np.allclose(output[1], expected_sentence_values)


def test_position_zero_has_alternating_sine_cosine():
    """Verify structural sanity: position 0 should alternate between 0 (sin) and 1 (cos)."""
    pe = PositionalEncoding(max_len=5, embedding_dim=4)

    # For pos=0, angle=0. sin(0)=0, cos(0)=1
    # Expected row 0 structure: [0.0, 1.0, 0.0, 1.0]
    expected_row_zero = np.array([0.0, 1.0, 0.0, 1.0])
    assert np.allclose(pe.encoding[0], expected_row_zero)
