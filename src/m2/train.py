import os
import pdb

import hydra
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf

import wandb
from m2.data import corrupt_mnist
from m2.model import MyAwesomeModel

load_dotenv()
LOGIN_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=LOGIN_KEY)

# PROJECT_PATH = os.path.dirname(__file__)
# CONFIG_PATH = os.path.abspath(os.path.join(PROJECT_PATH, os.pardir, os.pardir, 'configs'))


# NOTE: The config path should be relative to the root of the directory
# It is not here, and should ideally just be "configs"
@hydra.main(config_path="configs", config_name="default_config")
def train(cfg) -> None:
    """Train a model on MNIST."""
    # hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    HYDRA_PATH = os.getcwd()  # Hydra hijacks the cwd

    logger.remove()
    logger.add(os.path.join(HYDRA_PATH, "my_log.log"), level="DEBUG", rotation="100 MB")
    logger.info(cfg)

    logger.debug("Training day and night")
    train_params = cfg.train
    path_params = cfg.paths

    DATA_PATH = path_params["data"]
    MODEL_PATH = path_params["model"]
    FIGURE_PATH = path_params["figures"]

    run = wandb.init(
        entity="lenaribena-technical-university-of-denmark",
        project="mnist",
        name="experiment-2025-01-09",
        notes="Testing with more init data",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Load training set and create model
    train_set, _ = corrupt_mnist(data_path=DATA_PATH)
    input_dim = train_set[0][0].shape[1:]
    model = MyAwesomeModel(input_dim=input_dim)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=train_params["batch_size"], shuffle=True)

    # Training loop
    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(train_params["epochs"]):
        model.train()

        logger.debug("epoch: ", epoch + 1)
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
                logger.debug(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
                wandb.log({"Loss": loss.item()})

    logger.debug("Training complete")

    # Save the model and training statistics
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, "cnn.pth"))
    wandb.save(os.path.join(MODEL_PATH, "cnn.pth"))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"], alpha=0.5)
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"], alpha=0.5)
    axs[1].set_title("Train accuracy")
    fig.savefig(os.path.join(FIGURE_PATH, "training_statistics.png"))
    wandb.log({"plot": wandb.Image(fig)})

    return statistics


def main():
    _ = train()


if __name__ == "__main__":
    main()
