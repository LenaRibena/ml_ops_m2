import os

import matplotlib.pyplot as plt
import torch
import typer

from m2.data import corrupt_mnist
from m2.model import MyAwesomeModel

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3, epochs: int = 10, batch_size: int = 64, model_path: str = "models") -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here

    # Load training set and create model
    train_set, _ = corrupt_mnist()
    input_dim = train_set[0][0].shape[1:]
    model = MyAwesomeModel(input_dim=input_dim)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Training loop
    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()

        print("epoch: ", epoch + 1)
        for i, (images, targets) in enumerate(trainloader):
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, targets)
            loss.backward()
            optimizer.step()

            # Save the loss and accuracy
            statistics["train_loss"].append(loss.item())
            accuracy = (log_ps.argmax(dim=1) == targets).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")

    # Save the model and training statistics
    torch.save(model.state_dict(), f"{model_path}/cnn.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"], alpha=0.5)
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"], alpha=0.5)
    axs[1].set_title("Train accuracy")
    fig.savefig(os.path.join("reports", "training_statistics.png"))


def main():
    app()


if __name__ == "__main__":
    main()
