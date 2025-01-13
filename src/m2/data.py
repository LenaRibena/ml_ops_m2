import os
import torch
import typer


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""

    # Load training and test data
    train_images, train_target = [], []
    for i in range(6):
        train_path = os.path.join(raw_dir, f"train_images_{i}.pt")
        test_path = os.path.join(raw_dir, f"test_images_{i}.pt")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Raw data not found.")
        
        train_images.append(torch.load(train_path))
        train_target.append(torch.load(test_path))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(os.path.join(raw_dir, "test_images.pt"))
    test_target: torch.Tensor = torch.load(os.path.join(raw_dir, "test_target.pt"))

    # Add batch dimension to images and convert target to long
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    # Normalize images
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # Save processed data
    torch.save(train_images, os.path.join(processed_dir, "train_images.pt"))
    torch.save(train_target, os.path.join(processed_dir, "train_target.pt"))
    torch.save(test_images, os.path.join(processed_dir, "test_images.pt"))
    torch.save(test_target, os.path.join(processed_dir, "test_target.pt"))


def corrupt_mnist(data_path: str = os.path.join("data", "processed")) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    if not os.path.exists(data_path):
        raise FileNotFoundError("Processed data not found.")
    
    train_images = torch.load(os.path.join(data_path, "train_images.pt"))
    train_target = torch.load(os.path.join(data_path, "train_target.pt"))
    test_images = torch.load(os.path.join(data_path, "test_images.pt"))
    test_target = torch.load(os.path.join(data_path, "test_target.pt"))

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set

def main():
    typer.run(preprocess_data)

if __name__ == "__main__":
    main()
