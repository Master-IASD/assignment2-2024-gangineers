import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import torchvision.utils as vutils

from model import Generator, Discriminator
from utils import (D_train_WGAN_GP, G_train_WGAN_GP,
                   save_models, weights_init)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WGAN-GP.')
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('samples', exist_ok=True)

    # Data Pipeline
    print('Loading dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))])

    train_dataset = datasets.MNIST(root='data/MNIST/',
                                   train=True,
                                   transform=transform,
                                   download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    print('Dataset Loaded.')

    print('Loading models...')
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).cuda()
    D = Discriminator(d_input_dim=mnist_dim).cuda()

    # Apply weight initialization
    G.apply(weights_init)
    D.apply(weights_init)

    G = torch.nn.DataParallel(G)
    D = torch.nn.DataParallel(D)

    print('Models loaded.')

    # Optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
    D_optimizer = optim.Adam(D.parameters(), lr=3e-4, betas=(0.5, 0.9))

    # Training parameters
    lambda_gp = 10
    n_critic = 1  # Update discriminator and generator equally

    # Lists to keep track of progress
    D_losses = []
    G_losses = []

    print('Starting Training with WGAN-GP:')
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch + 1, leave=True):
        D_loss_epoch = 0.0
        G_loss_epoch = 0.0

        for i, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            batch_size = x.size(0)

            # Train Discriminator
            D_loss, T_real_mean, T_fake_mean = D_train_WGAN_GP(x, G, D, D_optimizer, lambda_gp)
            D_loss_epoch += D_loss

            # Optionally, print or log T_real_mean and T_fake_mean
            # print(f"Batch {i}: T_real_mean={T_real_mean:.4f}, T_fake_mean={T_fake_mean:.4f}")

            # Train Generator every n_critic steps
            if i % n_critic == 0:
                G_loss = G_train_WGAN_GP(G, D, G_optimizer, batch_size)
                G_loss_epoch += G_loss

        # Compute average losses for the epoch
        D_loss_epoch /= len(train_loader)
        G_loss_epoch /= len(train_loader) / n_critic

        D_losses.append(D_loss_epoch)
        G_losses.append(G_loss_epoch)

        # Save models every 10 epochs
        if epoch == 5 or epoch % 10 == 0:
            save_models(G, D, 'checkpoints_v2', prefix=f'WGAN_GP_{epoch}_')

            # Generate 5000 images and save them
            print(f'Generating images at epoch {epoch}...')
            os.makedirs(f'samples/{epoch}', exist_ok=True)
            G.eval()
            with torch.no_grad():
                n_samples = 0
                gen_batch_size = 100
                while n_samples < 3000:
                    z = torch.randn(gen_batch_size, 100).cuda()
                    x_fake = G(z).detach().cpu()
                    x_fake = x_fake.view(-1, 1, 28, 28)
                    for idx in range(x_fake.size(0)):
                        if n_samples >= 3000:
                            break
                        vutils.save_image(x_fake[idx],
                                          f'samples/{epoch}/{n_samples}.png',
                                          normalize=True)
                        n_samples += 1
            G.train()

        # Print losses every epoch
        print(f'Epoch [{epoch}/{n_epoch}] | '
              f'D Loss: {D_loss_epoch:.4f} | '
              f'G Loss: {G_loss_epoch:.4f}')

    print('Training done.')

    # Save the losses to JSON files
    with open('D_losses.json', 'w') as f:
        json.dump(D_losses, f)
    with open('G_losses.json', 'w') as f:
        json.dump(G_losses, f)

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('losses.png')
    plt.show()
