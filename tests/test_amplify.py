import torch

from src.amplification.amplify import amplify_activation


def test_amplify_activation_identity():
    z = torch.tensor([1.0, -2.0, 0.5])
    v = torch.tensor([0.5, 0.5, 0.5])
    out = amplify_activation(z, v, alpha=0.0, beta=0.01)
    assert torch.allclose(out, z)


def test_amplify_activation_expected():
    z = torch.tensor([3.0, 4.0])  # ||z|| = 5
    v = torch.tensor([0.0, 2.0])  # ||v|| = 2
    alpha = 0.5
    beta = 1.0
    scale = beta * (5.0 / 2.0)
    expected = (1 - alpha) * z + alpha * scale * v
    out = amplify_activation(z, v, alpha=alpha, beta=beta)
    assert torch.allclose(out, expected)
