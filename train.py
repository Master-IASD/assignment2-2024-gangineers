import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml


from model import Generator, Discriminator
from utils import D_train, G_train, save_models


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))

    wandb.init(
        project="gans-iasd",
        config={**config},
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

    parser = argparse.ArgumentParser(description="Train Normalizing Flow.")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="path to config file"
    )
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Data Pipeline
    print("Dataset loading...")
    # MNIST Dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))]
    )

    train_dataset = datasets.MNIST(
        root="data/MNIST/", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="data/MNIST/", train=False, transform=transform, download=False
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=config["batch_size"], shuffle=False
    )
    print("Dataset Loaded.")

    print("Model Loading...")
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    # if torch.cuda.is_available():
    # model = DataParallel(model).cuda()
    print("Model loaded.")

    # loss
    criterion = nn.BCELoss()

    # optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=config["lr_g"])
    D_optimizer = optim.Adam(D.parameters(), lr=config["lr_d"])

    print("Start Training :")

    n_epoch = config["n_epoch"]
    for epoch in trange(1, n_epoch + 1, leave=True):
        G_loss = 0.0
        D_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            D_loss += D_train(x, G, D, D_optimizer, criterion)
            G_loss += G_train(x, G, D, G_optimizer, criterion)

        D_loss /= len(train_loader.dataset)
        G_loss /= len(train_loader.dataset)
        wandb.log({"Discriminator Loss": D_loss, "Generator Loss": G_loss})

        if epoch % 10 == 0:
            save_models(
                G,
                D,
                model_file=f"epoch_{n_epoch}_lrG_{config["lr_g"]}_lrD_{config["lr_d"]}",
            )

    print("Training done")
