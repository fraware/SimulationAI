"""Tests for DQNModel (target network and output shape)."""

import numpy as np
import pytest

try:
    from inter_sim_rl.dqn_model import DQNModel

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_dqn_has_target_model():
    """DQNModel has target_model and update_target_model."""
    model = DQNModel(input_dim=24, output_dim=4)
    assert hasattr(model, "target_model")
    assert hasattr(model, "update_target_model")
    model.update_target_model()


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_dqn_output_shape():
    """model.predict returns (batch, output_dim)."""
    model = DQNModel(input_dim=24, output_dim=4)
    batch = np.random.randn(3, 24).astype(np.float32)
    out = model.model.predict(batch, verbose=0)
    assert out.shape == (3, 4)


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_update_target_copies_weights():
    """update_target_model copies weights from model to target_model."""
    model = DQNModel(input_dim=24, output_dim=4)
    # Change main model weights
    for w in model.model.weights:
        w.assign(w.numpy() + 1.0)
    model.update_target_model()
    for w_main, w_targ in zip(model.model.weights, model.target_model.weights):
        np.testing.assert_array_almost_equal(w_main.numpy(), w_targ.numpy())
