import torch 
import torchvision
import os

from model import Generator, Discriminator_WGAN_GP
from utils import transporter_in_latent
import torchvision.utils as vutils

if __name__ == '__main__':
    # Load the trained generator and discriminator
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).cuda()
    D = Discriminator_WGAN_GP(d_input_dim=mnist_dim).cuda()
    
    # Wrap models with DataParallel before loading state_dict
    G = torch.nn.DataParallel(G)
    D = torch.nn.DataParallel(D)


    model_wgan = 128
    # Load saved model weights
    G.load_state_dict(torch.load(f'checkpoints/WGAN_GP_{model_wgan}_G.pth', weights_only=True))
    D.load_state_dict(torch.load(f'checkpoints/WGAN_GP_{model_wgan}_D.pth', weights_only=True))


    # Parameters
    num_samples = 10000
    batch_size = 100

    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0

    with torch.no_grad():
            while n_samples<num_samples:
                z = torch.randn(batch_size, 100).cuda().requires_grad_(True)
                new_z = transporter_in_latent(G, D, z)
                x = G(new_z)
                x = x.reshape(batch_size, 28, 28)
                for k in range(x.shape[0]):
                    if n_samples<num_samples:
                        torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                        n_samples += 1




    
    