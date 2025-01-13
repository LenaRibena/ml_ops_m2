import os

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from m2.model import MyAwesomeModel
from src.m2.data import corrupt_mnist


def load_network(model_path: str) -> MyAwesomeModel:
    """Load a trained model from the specified model path.

    NB: This function assumes that the data is 28x28"""

    model = MyAwesomeModel((28, 28))
    model.load_state_dict(torch.load(model_path))
    return model


def extract_layer(model: MyAwesomeModel, layer_name: str) -> torch.Tensor:
    """Extract the second convolution layer from the model."""
    train, _ = corrupt_mnist()
    image, _ = next(iter(train))

    # Create a hook to extract the feature from the specified layer
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model.conv2.register_forward_hook(get_activation(layer_name))

    # Get the feature from the specified layer
    _ = model(image.unsqueeze(1))
    print(activation[layer_name])

    return activation[layer_name]


def visualize_features(feature: torch.Tensor, save_path: str = os.path.join("reports", "figures")) -> None:
    """Visualize the features (from extract_layer) using t-SNE."""

    # Reshape feature to comply with t-SNE requirements
    feature = feature.squeeze(0).view(feature.size(1), -1)  # (64, 14, 14) -> (64, 196)

    # Perform t-SNE and save the plot
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(feature)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)
    plt.savefig(os.path.join(save_path, "tsne_plot.png"))
    plt.close()


def main():
    model = load_network(os.path.join("models", "cnn.pth"))
    feature = extract_layer(model, "conv2")
    visualize_features(feature)


if __name__ == "__main__":
    main()
