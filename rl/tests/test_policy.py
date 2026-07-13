import numpy as np
from rl.policy import NumpyMLPPolicy


def test_forward_softmax_sums_to_one():
    w0 = np.eye(3); b0 = np.zeros(3)
    w1 = np.eye(3); b1 = np.zeros(3)
    pol = NumpyMLPPolicy([(w0, b0), (w1, b1)])
    out = pol(np.array([1.0, 2.0, 3.0]))
    assert abs(out.sum() - 1.0) < 1e-9
    assert (out >= 0).all()


def test_raw_action_matches_manual_two_layer():
    w0 = np.array([[1.0, 0.0], [0.0, 1.0]]); b0 = np.array([0.1, -0.1])
    w1 = np.array([[2.0, 0.0], [0.0, 2.0]]); b1 = np.array([0.0, 0.0])
    pol = NumpyMLPPolicy([(w0, b0), (w1, b1)])
    x = np.array([0.5, -0.5])
    h = np.tanh(w0 @ x + b0)     # hidden uses tanh
    expected = w1 @ h + b1       # last layer linear
    assert np.allclose(pol.raw_action(x), expected)


def test_from_npz_roundtrip(tmp_path):
    p = tmp_path / "m.npz"
    np.savez(p, w0=np.eye(2), b0=np.zeros(2), w1=np.eye(2), b1=np.zeros(2))
    pol = NumpyMLPPolicy.from_npz(str(p))
    out = pol(np.array([1.0, 2.0]))
    assert out.shape == (2,)
