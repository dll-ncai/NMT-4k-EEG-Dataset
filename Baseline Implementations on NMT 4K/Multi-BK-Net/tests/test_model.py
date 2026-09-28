import torch

from multibknet_nmt.model import MultiBKNet, count_trainable_parameters


def test_nmt19_model_shape_and_feature_length():
    model = MultiBKNet(n_channels=19)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 19, 6000))
    assert output.shape == (2, 2)
    assert model.final_feature_samples == 3
    assert count_trainable_parameters(model) > 1_000_000


def test_paper21_parameter_count():
    model = MultiBKNet(n_channels=21)
    assert count_trainable_parameters(model) == 1_038_683
