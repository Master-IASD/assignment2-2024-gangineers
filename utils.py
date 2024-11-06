import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from model import Discriminator

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

model = Discriminator(d_input_dim=784).to(device)
# Load the state dict and remove 'module.' prefix
state_dict = torch.load("checkpoints/D.pth", map_location=device)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
# Extract the weights from the first layer
weights = model.fc1.weight.data.cpu().numpy()


def plot_layer_weights(weights: np.ndarray, clip_value: float):
    heatmap_weights = weights.copy()

    # Apply a mask: set values > clip_value in absolute to a distinct color (e.g., 2 * clip_value)
    heatmap_weights[np.abs(weights) > clip_value] = (
        2 * clip_value
    )  # Use 2 * clip_value as a placeholder for red

    # Define the colormap with red for values exceeding clip_value
    cmap = sns.color_palette("viridis", as_cmap=True)
    cmap.set_over("red")  # Values greater than the color limit are shown in red

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_weights, cmap=cmap, cbar=True, vmax=clip_value)
    plt.title("Discriminator First Layer Weights with Clipping Highlighted")

    # Uncomment this line to log to wandb if needed
    wandb.log({"Discriminator First Layer Weights": wandb.Image(plt)})

    # plt.show()
    plt.close()


def D_train(x, G, D, D_optimizer, criterion):
    # =======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).to(device)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(device)

    D_output = D(x_fake)

    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion):
    # =======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)
    y = torch.ones(x.shape[0], 1).to(device)

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


def save_models(G, D, model_file: str, folder="checkpoints/"):
    torch.save(G.state_dict(), os.path.join(folder, "G_" + model_file))
    torch.save(D.state_dict(), os.path.join(folder, "D_" + model_file))


def load_model(G, folder="checkpoints/", model_file="G.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
    ckpt = torch.load(os.path.join(folder, model_file), map_location=device)
    G.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()})
    return G


if __name__ == "__main__":
    plot_layer_weights(weights, 0.01)
