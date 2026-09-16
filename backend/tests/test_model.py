"""Smoke tests for the siamese damage classifier.

Synthetic tensors only - these verify the network is wired correctly and can learn, without
needing the xBD download. Skipped if torch is not installed.
"""

import pytest

torch = pytest.importorskip("torch")

from ml.models.siamese import (  # noqa: E402
    DAMAGE_CLASSES,
    SiameseDamageNet,
    class_weights_from_counts,
)


@pytest.fixture(scope="module")
def model():
    return SiameseDamageNet(pretrained=False)


def test_forward_shape(model):
    pre = torch.randn(4, 3, 128, 128)
    post = torch.randn(4, 3, 128, 128)
    assert model(pre, post).shape == (4, len(DAMAGE_CLASSES))


def test_encoder_is_shared(model):
    """One encoder, two inputs - that is what makes it siamese. Two separate encoders would
    double the parameters and halve the data each one sees."""
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    total = sum(p.numel() for p in model.parameters())
    assert total < 2 * encoder_params


def test_swapping_pre_and_post_changes_the_prediction(model):
    """Damage is directional: intact -> rubble is not the same as rubble -> intact.
    The difference term in the head is what encodes that."""
    pre = torch.randn(2, 3, 128, 128)
    post = torch.randn(2, 3, 128, 128)
    assert not torch.allclose(model(pre, post), model(post, pre))


def test_predict_returns_classes_and_confidence(model):
    predicted, confidence = model.predict(torch.randn(3, 3, 128, 128), torch.randn(3, 3, 128, 128))
    assert predicted.shape == (3,)
    assert ((confidence >= 0.25) & (confidence <= 1.0)).all()


def test_class_weights_favour_the_rare_classes():
    """xBD is ~80% no-damage. Without inverse-frequency weighting the model predicts
    'fine' for everything."""
    weights = class_weights_from_counts(
        {"no-damage": 8000, "minor-damage": 800, "major-damage": 600, "destroyed": 400}
    )
    assert weights[0] < weights[1] < weights[2] < weights[3]


def test_model_can_overfit_a_tiny_batch():
    """If it cannot memorise four examples, training on 50,000 will not help either."""
    torch.manual_seed(0)
    net = SiameseDamageNet(pretrained=False)
    pre = torch.randn(4, 3, 64, 64)
    post = torch.randn(4, 3, 64, 64)
    labels = torch.tensor([0, 1, 2, 3])

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    first = criterion(net(pre, post), labels).item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = criterion(net(pre, post), labels)
        loss.backward()
        optimizer.step()

    assert loss.item() < first * 0.5
