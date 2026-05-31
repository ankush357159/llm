import numpy as np
import unittest


class SelfAttention:
    """
    A single-head Self-Attention mechanism implemented from scratch using NumPy.

    This class computes scaled dot-product attention for a single input sequence.
    It projects the input matrix into Query (Q), Key (K), and Value (V) matrices
    using learned weights, calculates attention scores, applies softmax normalization,
    and returns a weighted representation of the input.

    Attributes:
        embedding_dim (int): The dimensionality of the input and output embeddings.
        W_q (numpy.ndarray): Weight matrix for the Query projection. Shape: (d, d).
        W_k (numpy.ndarray): Weight matrix for the Key projection. Shape: (d, d).
        W_v (numpy.ndarray): Weight matrix for the Value projection. Shape: (d, d).
    """

    def __init__(self, embedding_dim):
        """
        Initializes the SelfAttention mechanism with random weights.

        Args:
            embedding_dim (int): The dimensionality of the token embeddings.
        """
        self.embedding_dim = embedding_dim
        self.W_q = np.random.randn(embedding_dim, embedding_dim) * 0.01
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * 0.01
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * 0.01

    def softmax(self, x):
        """
        Computes the softmax activation along the last axis.

        Args:
            x (numpy.ndarray): Input tensor of any shape.

        Returns:
            numpy.ndarray: Softmax probabilities with the same shape as x.
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Safe softmax subtraction
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x):
        """
        Executes the forward pass of the self-attention layer.

        Args:
            x (numpy.ndarray): Input sequence tensor. Shape: (sequence_length, embedding_dim).

        Returns:
            numpy.ndarray: Attention-weighted output tensor. Shape: (sequence_length, embedding_dim).
        """
        # x shape: (sequence_length, embedding_dim)
        Q = x @ self.W_q  # (sequence_length, embedding_dim)
        K = x @ self.W_k  # (sequence_length, embedding_dim)
        V = x @ self.W_v  # (sequence_length, embedding_dim)

        # Compute attention scores
        attention_scores = Q @ K.T / np.sqrt(self.embedding_dim)  # (sequence_length, sequence_length)
        attention_weights = self.softmax(attention_scores)  # (sequence_length, sequence_length)

        # Compute the output as a weighted sum of values
        output = attention_weights @ V  # (sequence_length, embedding_dim)
        return output, Q, K, V, attention_scores, attention_weights