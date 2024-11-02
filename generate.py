import torch
import matplotlib.pyplot as plt
import torchvision
import os
import argparse
from tqdm import tqdm

from model import Generator
from utils import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Normalizing Flow.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="The batch size to use for training.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, randomly display generated images.",
    )
    args = parser.parse_args()

    print("Model Loading...")
    # Model Pipeline
    mnist_dim = 784

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = Generator(g_output_dim=mnist_dim).to(device)
    model = load_model(model, "checkpoints", "W_G.pth")
    model.eval()

    print("Model loaded.")

    print("Start Generating")
    os.makedirs("samples", exist_ok=True)

    total_samples = 10000
    batch_size = args.batch_size

    with torch.no_grad():
        for n_samples in tqdm(range(0, total_samples, batch_size)):
            z = torch.randn(batch_size, 100, device=device)
            x = model(z)
            x = x.reshape(batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples + k < total_samples:
                    torchvision.utils.save_image(
                        x[k : k + 1], os.path.join("samples", f"{n_samples + k}.png")
                    )

    if args.plot:

        plt.figure(figsize=(10, 10))
        random_indices = torch.randint(0, total_samples, (25,))
        for i in range(25):
            img = plt.imread(f"samples/{i}.png")
            plt.subplot(5, 5, i + 1)
            plt.imshow(img, cmap="gray")
            plt.axis("off")
        plt.show()
