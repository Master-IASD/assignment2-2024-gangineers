import wandb
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Critic, Generator
from torchvision import datasets, transforms
import yaml
import matplotlib

matplotlib.use("Agg")


##################DATA PIPELINE##################
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
    dataset=train_dataset, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=False
)
print("Dataset Loaded.")

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
##################DATA PIPELINE##################

wandb.login()

with open("config_sweep.yaml") as file:
    sweep_configuration = yaml.safe_load(file)


# 1: Define objective/training function
def train(config):
    # Access sweep parameters from wandb.config directly
    n_epochs = config.n_epochs
    n_critic = config.n_critic
    clip_value = config.clip_value
    batch_size = config.batch_size
    z_dim = config.z_dim

    # Update DataLoader with current batch_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    model_G = Generator(g_output_dim=784).to(device)
    model_C = Critic().to(device)

    optimizer_G = torch.optim.RMSprop(model_G.parameters(), lr=1e-4)
    optimizer_C = torch.optim.RMSprop(model_C.parameters(), lr=1e-4)

    for epoch in tqdm(range(n_epochs)):
        for i, (real_images, _) in enumerate(train_loader):
            real_images = real_images.view(-1, 784).to(device)  # Flatten MNIST images

            # Train Critic
            for _ in range(n_critic):
                optimizer_C.zero_grad()
                z = torch.randn(batch_size, z_dim).to(device)
                fake_images = model_G(z).detach()
                real_validity = model_C(real_images)
                fake_validity = model_C(fake_images)
                loss_C = -(torch.mean(real_validity) - torch.mean(fake_validity))
                loss_C.backward()
                optimizer_C.step()

                for p in model_C.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, z_dim).to(device)
            gen_images = model_G(z)
            gen_validity = model_C(gen_images)
            loss_G = -torch.mean(gen_validity)
            loss_G.backward()
            optimizer_G.step()

            # Log Progress
            if i % 100 == 0:
                print(
                    f"Epoch [{epoch}/{n_epochs}] Batch {i}/{len(train_loader)} "
                    f"Loss C: {loss_C.item():.4f}, Loss G: {loss_G.item():.4f}"
                )
                wandb.log(
                    {"Loss Critic": loss_C.item(), "Loss Generator": loss_G.item()}
                )

    # Log first layer weights heatmap
    first_layer_weights = list(model_C.parameters())[0].detach().cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(first_layer_weights, cmap="viridis", vmin=-clip_value, vmax=clip_value)
    plt.title("Discriminator First Layer Weights")
    wandb.log({"Discriminator First Layer Weights": wandb.Image(plt)})
    plt.close()

    # Save models
    torch.save(model_G.state_dict(), "checkpoints/W_G_trash.pth")
    torch.save(model_C.state_dict(), "checkpoints/W_C_trash.pth")


def main():
    wandb.init(project="my-first-sweep")
    train(wandb.config)


# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
