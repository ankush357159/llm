import pytest
import numpy as np
from embedding.LayerNorm import LayerNorm


def test_layer_norm_output_shape():
    """Verify LayerNorm preserves the exact shape of the input tensor."""
    embedding_dim = 4
    ln = LayerNorm(embedding_dim)

    # 3D Batch shape: (batch_size=2, seq_len=3, embedding_dim=4)
    x = np.random.randn(2, 3, embedding_dim)
    output = ln.forward(x)

    assert output.shape == (2, 3, embedding_dim)


def test_layer_norm_zero_mean_unit_variance():
    """Verify that features have a mean of 0 and variance of 1 after normalization."""
    embedding_dim = 8
    ln = LayerNorm(embedding_dim)

    # Generate arbitrary random vectors
    x = np.random.randn(4, 5, embedding_dim) * 5.0 + 10.0
    output = ln.forward(x)

    # Calculate output mean and variance across the last dimension
    out_mean = np.mean(output, axis=-1)
    out_var = np.var(output, axis=-1)

    # Mean should be very close to 0, and variance very close to 1
    assert np.allclose(out_mean, 0.0, atol=1e-6)
    assert np.allclose(out_var, 1.0, atol=1e-6)


def test_layer_norm_with_custom_gamma_beta():
    """Verify that output adapts correctly when gamma and beta values change."""
    embedding_dim = 3
    ln = LayerNorm(embedding_dim)

    # Alter the weights manually to simulate learned weights
    ln.gamma = np.array([2.0, 2.0, 2.0])
    ln.beta = np.array([0.5, 0.5, 0.5])

    x = np.array([[1.0, 2.0, 3.0]])  # 2D Single sentence slice
    output = ln.forward(x)

    # For [1, 2, 3]: mean=2.0, variance=0.66666667
    # Normalized: [-1.2247, 0.0, 1.2247]
    # Scaled (x2 + 0.5): [-1.9494, 0.5, 2.9494]
    expected = np.array([[-1.94948974, 0.5, 2.94948974]])
    assert np.allclose(output, expected, atol=1e-6)
