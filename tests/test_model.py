import pytest
from tts.model import Encoder, TTS
from tts.config import Config
import numpy as np
import torch

@pytest.fixture
def X_fixture():
    rng = np.random.default_rng(0)
    return torch.from_numpy(rng.random((10,3))).float()

def test_encoder_shape(X_fixture):
    X = X_fixture
    config = Config(n_features=3,n_basis=7)
    encoder = Encoder(config)
    assert encoder(X).shape == (10, 7)

def test_tts_shape(X_fixture):
    X = X_fixture
    config = Config(n_features=3,n_basis=7)
    tts = TTS(config)
    rng = np.random.default_rng(0)
    Ns = rng.integers(1, 20, 10)
    Phis = [torch.from_numpy(rng.random((N, 7))).float() for N in Ns]
    result = tts(X, Phis)
    assert len(result) == 10
    for d in range(10):
        assert result[d].shape == (Ns[d])






