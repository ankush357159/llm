import numpy as np
import numpy.typing as npt
from typing import cast

from embedding.FeedForward import FeedForward, FeeedForward


Array = npt.NDArray[np.float64]


def test_forward_output_shape() -> None:
    """Forward pass preserves batch/sequence dimensions and embedding width."""
    embedding_dim = 8
    hidden_dim = 16
    seq_len = 10

    ff = FeedForward(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    x: Array = cast(Array, np.random.randn(seq_len, embedding_dim))
    y = ff.forward(x)

    assert y.shape == (seq_len, embedding_dim)


def test_relu_activation_behavior_with_identity_weights() -> None:
    """With identity projections and zero bias, output equals ReLU(input)."""
    embedding_dim = 4
    hidden_dim = 4

    ff = FeedForward(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    ff.W1 = np.eye(embedding_dim)
    ff.W2 = np.eye(embedding_dim)
    ff.b1 = np.zeros(hidden_dim)
    ff.b2 = np.zeros(embedding_dim)

    x: Array = np.array([
        [-1.0, 0.5, -2.0, 3.0],
        [0.0, -0.1, 1.5, -4.0],
    ])

    y = ff.forward(x)
    expected = np.maximum(0, x)
    np.testing.assert_allclose(y, expected, rtol=1e-7, atol=1e-7)


def test_feedforward_alias_works() -> None:
    """Misspelled legacy class name remains usable via alias."""
    model = FeeedForward(embedding_dim=6, hidden_dim=12)
    x: Array = cast(Array, np.random.randn(3, 6))
    y = model.forward(x)

    assert y.shape == (3, 6)


