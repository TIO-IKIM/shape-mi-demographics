"""Tests for the deep models (PointNet/DGCNN multi-organ multi-task).

Skipped automatically if torch is not installed (it is an optional dependency)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from shapedem.models.multitask import MultiOrganMultiTask, PointNetEncoder, DGCNNEncoder, TASKS


# Verifies PointNet maps (batch=5, 128 points, 3 coords) to (5, 32) embeddings; wrong shape means the architecture is misconfigured.
def test_pointnet_encoder_shape():
    enc = PointNetEncoder(emb=32)
    assert enc(torch.randn(5, 128, 3)).shape == (5, 32)


# Same contract as PointNet: DGCNN must also produce (batch, emb) embeddings from raw point clouds.
def test_dgcnn_encoder_shape():
    enc = DGCNNEncoder(emb=32, k=8)
    assert enc(torch.randn(4, 64, 3)).shape == (4, 32)


# mask[0,:]=0 simulates a subject with all organs missing; the model must still produce finite (NaN-free) output for all tasks.
@pytest.mark.parametrize("encoder", ["pointnet", "dgcnn"])
def test_multiorgan_forward_handles_missing_organs(encoder):
    model = MultiOrganMultiTask(n_organs=3, emb=16, encoder=encoder)
    B, O, N = 2, 3, 64
    pts = torch.randn(B, O, N, 3)
    size = torch.randn(B, O)
    mask = torch.ones(B, O)
    mask[0, :] = 0.0                       # subject 0 has NO organs -> must not produce NaN
    out = model(pts, size, mask)
    for t in TASKS:
        assert out[t].shape == (B,)
        assert torch.isfinite(out[t]).all()


# Sanity check: overfitting a tiny batch for 40 steps must reduce loss by at least 20%; failure means gradients do not flow correctly.
def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model = MultiOrganMultiTask(n_organs=2, emb=16, encoder="pointnet")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    B, O, N = 16, 2, 64
    pts = torch.randn(B, O, N, 3)
    size = torch.randn(B, O)
    mask = torch.ones(B, O)
    y = (pts.mean(dim=(1, 2, 3)) > 0).float()    # learnable function of the input
    bce = torch.nn.BCEWithLogitsLoss()
    first = last = None
    for step in range(40):
        out = model(pts, size, mask)
        loss = bce(out["sex"], y)
        val = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0:
            first = val
        last = val
    assert last < first * 0.8                     # the model fits the batch
