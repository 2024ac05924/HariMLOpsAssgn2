import torch

from src.model import SimpleCNN, predict_class


def test_model_output_shape():
    model = SimpleCNN()

    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    assert tuple(output.shape) == (2, 2)


def test_predict_class_returns_probabilities():
    model = SimpleCNN()

    x = torch.randn(1, 3, 224, 224)
    predicted_class, probabilities = predict_class(model, x)

    assert predicted_class.shape == (1,)
    assert probabilities.shape == (1, 2)

    probability_sum = probabilities.sum(dim=1)

    assert torch.allclose(
        probability_sum,
        torch.ones(1),
        atol=1e-6,
    )
