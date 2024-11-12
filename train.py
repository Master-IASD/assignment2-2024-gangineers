import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import torchvision.utils as vutils

from model import Generator, Discriminator, Discriminator_WGAN_GP
from utils import (D_train_WGAN_GP, G_train_WGAN_GP,
                   save_models, weights_init)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train WGAN-GP.')
#     parser.add_argument("--epochs", type=int, default=200,
#                         help="Number of epochs for training.")
#     parser.add_argument("--batch_size", type=int, default=64,
#                         help="Size of mini-batches for SGD")

#     args = parser.parse_args()

#     os.makedirs('checkpoints', exist_ok=True)
#     os.makedirs('data', exist_ok=True)
#     os.makedirs('samples', exist_ok=True)

#     # Data Pipeline
#     print('Loading dataset...')
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5,), std=(0.5,))])

#     train_dataset = datasets.MNIST(root='data/MNIST/',
#                                    train=True,
#                                    transform=transform,
#                                    download=True)

#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                batch_size=args.batch_size,
#                                                shuffle=True)
#     print('Dataset Loaded.')

#     print('Loading models...')
#     mnist_dim = 784
#     G = Generator(g_output_dim=mnist_dim).cuda()
#     D = Discriminator(d_input_dim=mnist_dim).cuda()

#     # Apply weight initialization
#     G.apply(weights_init)
#     D.apply(weights_init)

#     G = torch.nn.DataParallel(G)
#     D = torch.nn.DataParallel(D)

#     print('Models loaded.')

#     # Optimizers
#     G_optimizer = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
#     D_optimizer = optim.Adam(D.parameters(), lr=3e-4, betas=(0.5, 0.9))

#     # Training parameters
#     lambda_gp = 10
#     n_critic = 1  # Update discriminator and generator equally

#     # Lists to keep track of progress
#     D_losses = []
#     G_losses = []

#     print('Starting Training with WGAN-GP:')
#     n_epoch = args.epochs
#     for epoch in trange(1, n_epoch + 1, leave=True):
#         D_loss_epoch = 0.0
#         G_loss_epoch = 0.0

#         for i, (x, _) in enumerate(train_loader):
#             x = x.view(-1, mnist_dim)
#             batch_size = x.size(0)

#             # Train Discriminator
#             D_loss, T_real_mean, T_fake_mean = D_train_WGAN_GP(x, G, D, D_optimizer, lambda_gp)
#             D_loss_epoch += D_loss

#             # Optionally, print or log T_real_mean and T_fake_mean
#             # print(f"Batch {i}: T_real_mean={T_real_mean:.4f}, T_fake_mean={T_fake_mean:.4f}")

#             # Train Generator every n_critic steps
#             if i % n_critic == 0:
#                 G_loss = G_train_WGAN_GP(G, D, G_optimizer, batch_size)
#                 G_loss_epoch += G_loss

#         # Compute average losses for the epoch
#         D_loss_epoch /= len(train_loader)
#         G_loss_epoch /= len(train_loader) / n_critic

#         D_losses.append(D_loss_epoch)
#         G_losses.append(G_loss_epoch)

#         # Save models every 10 epochs
#         if epoch == 5 or epoch % 10 == 0:
#             save_models(G, D, 'checkpoints_v2', prefix=f'WGAN_GP_{epoch}_')

#             # Generate 5000 images and save them
#             print(f'Generating images at epoch {epoch}...')
#             os.makedirs(f'samples/{epoch}', exist_ok=True)
#             G.eval()
#             with torch.no_grad():
#                 n_samples = 0
#                 gen_batch_size = 100
#                 while n_samples < 3000:
#                     z = torch.randn(gen_batch_size, 100).cuda()
#                     x_fake = G(z).detach().cpu()
#                     x_fake = x_fake.view(-1, 1, 28, 28)
#                     for idx in range(x_fake.size(0)):
#                         if n_samples >= 3000:
#                             break
#                         vutils.save_image(x_fake[idx],
#                                           f'samples/{epoch}/{n_samples}.png',
#                                           normalize=True)
#                         n_samples += 1
#             G.train()

#         # Print losses every epoch
#         print(f'Epoch [{epoch}/{n_epoch}] | '
#               f'D Loss: {D_loss_epoch:.4f} | '
#               f'G Loss: {G_loss_epoch:.4f}')

#     print('Training done.')

#     # Save the losses to JSON files
#     with open('D_losses.json', 'w') as f:
#         json.dump(D_losses, f)
#     with open('G_losses.json', 'w') as f:
#         json.dump(G_losses, f)

#     # Plot the losses
#     plt.figure(figsize=(10, 5))
#     plt.title("Generator and Discriminator Loss During Training")
#     plt.plot(G_losses, label="G")
#     plt.plot(D_losses, label="D")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.savefig('losses.png')
#     plt.show()


# New training loop starting from epoch 110 to 130
def new_train():
    parser = argparse.ArgumentParser(description='Resume Training WGAN-GP from Epoch 110 to 130.')
    parser.add_argument("--start_epoch", type=int, default=110,
                        help="Starting epoch number.")
    parser.add_argument("--end_epoch", type=int, default=130,
                        help="Ending epoch number.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Size of mini-batches for SGD")
    parser.add_argument("--checkpoint_folder", type=str, default='checkpoints_v3',
                        help="Folder where checkpoints are saved.")
    parser.add_argument("--sample_folder", type=str, default='samples_v3',
                        help="Folder where samples will be saved.")
    parser.add_argument("--lambda_gp", type=float, default=10,
                        help="Gradient penalty coefficient.")
    parser.add_argument("--n_critic", type=int, default=1,
                        help="Number of discriminator updates per generator update.")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_folder, exist_ok=True)
    os.makedirs(args.sample_folder, exist_ok=True)

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
    D = Discriminator_WGAN_GP(d_input_dim=mnist_dim).cuda()

    # Apply weight initialization
    G.apply(weights_init)
    D.apply(weights_init)

    G = torch.nn.DataParallel(G)
    D = torch.nn.DataParallel(D)

    print('Loading checkpoint...')
    checkpoint_epoch = args.start_epoch
    G_checkpoint_path = os.path.join("checkpoints_v2", f'WGAN_GP_{checkpoint_epoch}_G.pth')
    D_checkpoint_path = os.path.join("checkpoints_v2", f'WGAN_GP_{checkpoint_epoch}_D.pth')

    # Load saved model weights
    G.load_state_dict(torch.load(G_checkpoint_path, map_location='cuda'))
    D.load_state_dict(torch.load(D_checkpoint_path, map_location='cuda'))
    print(f'Loaded Generator from {G_checkpoint_path}')
    print(f'Loaded Discriminator from {D_checkpoint_path}')

    # Optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))
    D_optimizer = optim.Adam(D.parameters(), lr=1.5e-4, betas=(0.5, 0.9))

    # Optionally, load optimizer states if available
    # Example:
    # G_optimizer.load_state_dict(torch.load(os.path.join(args.checkpoint_folder, f'WGAN_GP_{checkpoint_epoch}_G_opt.pth')))
    # D_optimizer.load_state_dict(torch.load(os.path.join(args.checkpoint_folder, f'WGAN_GP_{checkpoint_epoch}_D_opt.pth')))

    # Training parameters
    lambda_gp = args.lambda_gp
    n_critic = args.n_critic  # Update discriminator and generator equally

    # Lists to keep track of progress
    D_losses = []
    G_losses = []

    # print args
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    print(f'Starting Training from Epoch {args.start_epoch +1} to {args.end_epoch} with WGAN-GP:')
    for epoch in trange(args.start_epoch + 1, args.end_epoch + 1, leave=True):
        D_loss_epoch = 0.0
        G_loss_epoch = 0.0

        for i, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).cuda()
            batch_size = x.size(0)

            # Train Discriminator
            D_loss, T_real_mean, T_fake_mean = D_train_WGAN_GP(x, G, D, D_optimizer, lambda_gp)
            D_loss_epoch += D_loss

            # Train Generator every n_critic steps
            if i % n_critic == 0:
                G_loss = G_train_WGAN_GP(G, D, G_optimizer, batch_size)
                G_loss_epoch += G_loss

        # Compute average losses for the epoch
        D_loss_epoch /= len(train_loader)
        G_loss_epoch /= len(train_loader) / n_critic

        D_losses.append(D_loss_epoch)
        G_losses.append(G_loss_epoch)

        # Save models every epoch
        save_models(G, D, args.checkpoint_folder, prefix=f'WGAN_GP_{epoch}_')

        # Generate 5000 images and save them
        print(f'Generating 5000 images at epoch {epoch}...')
        os.makedirs(f'{args.sample_folder}/{epoch}', exist_ok=True)
        G.eval()
        with torch.no_grad():
            n_samples = 0
            gen_batch_size = 100
            num_samples = 5000  # Set to 5000 as per requirement
            while n_samples < num_samples:
                z = torch.randn(gen_batch_size, 100).cuda()
                x_fake = G(z).detach().cpu()
                x_fake = x_fake.view(-1, 1, 28, 28)
                for idx in range(x_fake.size(0)):
                    if n_samples >= num_samples:
                        break
                    vutils.save_image(x_fake[idx],
                                      os.path.join(args.sample_folder, f'{epoch}', f'{n_samples}.png'),
                                      normalize=True)
                    n_samples += 1
        G.train()

        # Print losses every epoch
        print(f'Epoch [{epoch}/{args.end_epoch}] | '
              f'D Loss: {D_loss_epoch:.4f} | '
              f'G Loss: {G_loss_epoch:.4f}')

    print('Training done.')

    # Save the losses to JSON files
    with open('D_losses_resume.json', 'w') as f:
        json.dump(D_losses, f)
    with open('G_losses_resume.json', 'w') as f:
        json.dump(G_losses, f)

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training (Resume)")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('losses_resume.png')
    plt.show()

if __name__ == '__main__':
    new_train()