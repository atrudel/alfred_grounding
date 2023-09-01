import pytest
import torch

from models.clasp import CLASP

Z_SIZE = 4

@pytest.fixture(scope="session")
def clasp_model():
    # Instantiate your object here
    clasp: CLASP = CLASP(z_size=Z_SIZE)
    yield clasp


def test_contrastive_loss_should_return_zero_with_identical_zs(clasp_model):
    # Given
    z_text = torch.tensor([[1,0,0,0],
                           [0,0,1,0],
                           [0,1,0,0]], dtype=float)
    z_behavior = z_text.clone()

    # When
    loss = clasp_model.contrastive_loss(z_text, z_behavior)

    # Then
    assert loss.item() == pytest.approx(0, abs=1e-5)


def test_contrastive_loss_should_be_high_with_orthogonal_z(clasp_model):
    # Given
    z_size = 8
    z_text = torch.tensor([[1, 0, 0, 0 ],
                           [0, 0, 1, 0]], dtype=float)
    z_behavior = torch.tensor([[0, 1, 0, 0 ],
                               [0, 0, 0, 1]], dtype=float)

    # When
    loss = clasp_model.contrastive_loss(z_text, z_behavior)

    # Then
    assert loss > 0.5