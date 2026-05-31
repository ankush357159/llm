import  pytest
from embedding.SelfAttention import SelfAttention
import numpy as np

@pytest.fixture
def sample_setup():
    """Fixture providing standard dimension sizes for the tests."""
    return {"embedding_dim": 64, "seq_len": 10}


@pytest.mark.parametrize("seq_len, embedding_dim", [
    (1, 16),
    (10, 64),
    (32, 128)
])
def test_output_shape(seq_len, embedding_dim):
    """Verify that the output tensor maintains the correct dimensions across varied inputs."""
    attention = SelfAttention(embedding_dim)
    dummy_input = np.random.randn(seq_len, embedding_dim)

    output = attention.forward(dummy_input)

    assert output.shape == (seq_len, embedding_dim)


def test_softmax_row_sum(sample_setup):
    """Verify that the rows of the generated attention weights matrix sum exactly to 1.0."""
    emb_dim = sample_setup["embedding_dim"]
    seq_len = sample_setup["seq_len"]

    attention = SelfAttention(emb_dim)
    dummy_input = np.random.randn(seq_len, emb_dim)

    # Isolate internal scores to evaluate softmax independently
    Q = dummy_input @ attention.W_q
    K = dummy_input @ attention.W_k
    scores = Q @ K.T / np.sqrt(emb_dim)
    weights = attention.softmax(scores)

    row_sums = np.sum(weights, axis=-1)

    # Assert row values sum up closely to 1.0 within structural tolerances
    np.testing.assert_allclose(row_sums, np.ones(seq_len), rtol=1e-6)


def test_softmax_numerical_stability():
    """Verify that the softmax method handles extreme input values without encountering NaN errors."""
    attention = SelfAttention(embedding_dim=4)
    extreme_input = np.array([[1000.0, 1000.0, 1000.0], [-1000.0, -1000.0, -1000.0]])

    probabilities = attention.softmax(extreme_input)

    assert not np.isnan(probabilities).any()
    assert not np.isinf(probabilities).any()


def test_identity_weights_passthrough():
    """Verify system output integrity using static Identity matrix transforms."""
    dim = 4
    seq = 3
    attention = SelfAttention(dim)

    # Inject Identity matrices manually
    attention.W_q = np.eye(dim)
    attention.W_k = np.eye(dim)
    attention.W_v = np.eye(dim)

    # Define an orthogonal setup matrix
    input_data = np.eye(dim)[:seq]  # Matrix shape: (3, 4)

    output = attention.forward(input_data)

    assert output.shape == (seq, dim)
    assert not np.isnan(output).any()