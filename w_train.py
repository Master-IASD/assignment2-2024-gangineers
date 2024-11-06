import torch
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Critic, Generator
from torchvision import datasets, transforms
import yaml

config = yaml.safe_load(open("config_wgan.yaml"))

wandb.init(
    project="gans-iasd",
    config={**config},
)

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

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
model_G = Generator(g_output_dim=784).to(device)
model_C = Critic().to(device)

optimizer_G = torch.optim.RMSprop(model_G.parameters(), lr=1e-4)
optimizer_C = torch.optim.RMSprop(model_C.parameters(), lr=1e-4)

for epoch in tqdm(range(config["n_epochs"])):
    for i, (real_images, _) in enumerate(train_loader):
        # Move real images to device
        real_images = real_images.view(-1, 784).to(device)  # Flatten MNIST images

        # ---------------------
        #  Train Critic
        # ---------------------
        for _ in range(config["n_critic"]):
            # Zero the gradients on the Critic
            optimizer_C.zero_grad()

            # Generate fake images
            z = torch.randn(config["batch_size"], config["z_dim"]).to(device)
            fake_images = model_G(
                z
            ).detach()  # Detach to avoid gradient computation for Generator

            # Compute Critic outputs
            real_validity = model_C(real_images)
            fake_validity = model_C(fake_images)

            # Compute Wasserstein loss for Critic
            loss_C = -(torch.mean(real_validity) - torch.mean(fake_validity))

            # Backward and optimize
            loss_C.backward()
            optimizer_C.step()

            # Weight clipping
            for p in model_C.parameters():
                p.data.clamp_(-config["clip_value"], config["clip_value"])

        # ---------------------
        #  Train Generator
        # ---------------------
        # Zero the gradients on the Generator
        optimizer_G.zero_grad()

        # Generate fake images
        z = torch.randn(config["batch_size"], config["z_dim"]).to(device)
        gen_images = model_G(z)

        # Compute Critic output on generated images
        gen_validity = model_C(gen_images)

        # Compute Generator loss (to maximize the Critic's estimate)
        loss_G = -torch.mean(gen_validity)

        # Backward and optimize
        loss_G.backward()
        optimizer_G.step()

        # ---------------
        # Log Progress
        # ---------------
        if i % 100 == 0:
            print(
                f"Epoch [{epoch}/{config["n_epochs"]}] Batch {i}/{len(train_loader)} \
                  Loss C: {loss_C.item():.4f}, Loss G: {loss_G.item():.4f}"
            )

first_layer_weights = list(model_C.parameters())[0].detach().cpu().numpy()

# Plot heatmap of the weights
plt.figure(figsize=(10, 8))
sns.heatmap(first_layer_weights, cmap="viridis")
plt.title("Discriminator First Layer Weights")

# Log to wandb
wandb.log(
    {"Discriminator First Layer Weights": wandb.Image(plt)}, step=config["n_epochs"]
)

# Clear the plot
plt.close()

# Save models
torch.save(model_G.state_dict(), "checkpoints/W_G_trash.pth")
torch.save(model_C.state_dict(), "checkpoints/W_C_trash.pth")
