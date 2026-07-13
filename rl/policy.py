"""Serve-time policy inference. numpy only (no torch)."""
import numpy as np


def _softmax(a):
    a = np.asarray(a, float)
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


class NumpyMLPPolicy:
    def __init__(self, layers):
        self.layers = [(np.asarray(w, float), np.asarray(b, float)) for w, b in layers]

    @classmethod
    def from_npz(cls, path):
        data = np.load(path)
        layers = []
        i = 0
        while f"w{i}" in data:
            layers.append((data[f"w{i}"], data[f"b{i}"]))
            i += 1
        return cls(layers)

    def raw_action(self, obs):
        x = np.asarray(obs, float)
        for k, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if k < len(self.layers) - 1:
                x = np.tanh(x)
        return x

    def __call__(self, obs):
        return _softmax(self.raw_action(obs))
