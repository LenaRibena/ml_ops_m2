import re

import pytest
import torch

from m2.model import MyAwesomeModel


def test_my_model():
    # Test the model output shape
    model = MyAwesomeModel((28, 28))
    x = torch.randn(1, 1, 28, 28)
    y = model(x)

    assert y.shape == (1, 10), "The output shape is incorrect."


def test_model_warnings():
    model = MyAwesomeModel((28, 28))

    with pytest.raises(ValueError, match="Input tensor must have 4 dimensions."):
        x = torch.randn(1, 1, 26)
        _ = model(x)

    with pytest.raises(ValueError, match=re.escape("Input tensor must have shape (batch_size, 1, 28, 28).")):
        x = torch.randn(1, 3, 26, 28)
        _ = model(x)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_model_batch_size(batch_size):
    model = MyAwesomeModel((28, 28))
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)

    assert y.shape == (batch_size, 10), "The output shape is incorrect."
