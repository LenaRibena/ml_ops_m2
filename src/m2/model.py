import torch
from torch import nn

from data import corrupt_mnist


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, input_dim: tuple[int, int], dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * input_dim[0] // 4 * input_dim[1] // 4, 128)
        self.fc2 = nn.Linear(128, 10)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.dropout(self.activation(self.conv1(x)))
        x = self.maxpool(x)
        x = self.dropout(self.activation(self.conv2(x)))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.activation(self.fc1(x)))

        return self.logsoft(self.fc2(x))


if __name__ == "__main__":
    # Load the data
    train, _ = corrupt_mnist()
    images, targets = next(iter(train))

    # Create the model
    model = MyAwesomeModel((28, 28))

    # Pass an image (with artificial batch size added)
    log_ps = model(images.unsqueeze(1))

    # Print the model and the number of parameters
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
