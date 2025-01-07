import torch
import typer
from .model import MyAwesomeModel

from .data import corrupt_mnist

app = typer.Typer()


@app.command()
def evaluate(model_checkpoint: str, batch_size: int = 64) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here

    # Load dataset and model
    _, test_set = corrupt_mnist()
    input_dim = test_set[0][0].shape[1:]
    model = MyAwesomeModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_checkpoint))

    model.eval()
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    correct, total = 0, 0

    with torch.no_grad():
        # Iterate over test set and count correct predictions
        for images, targets in testloader:
            log_ps = model(images)

            correct += (log_ps.argmax(dim=1) == targets).float().sum().item()
            total += targets.size(0)

    print(f"Test accuracy: {correct / total}")

def main():
    app()

if __name__ == "__main__":
    main()
