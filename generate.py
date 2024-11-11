# # import torch 
# # import torchvision
# # import os
# # import argparse

# # from model import Generator, Discriminator_WGAN_GP
# # from utils import load_model, generate_samples_with_DRS
# # import torchvision.utils as vutils

# # if __name__ == '__main__':
# #     # Load the trained generator and discriminator
# #     mnist_dim = 784
# #     G = Generator(g_output_dim=mnist_dim).cuda()
# #     D = Discriminator_WGAN_GP(d_input_dim=mnist_dim).cuda()
    
# #     # Wrap models with DataParallel before loading state_dict
# #     G = torch.nn.DataParallel(G)
# #     D = torch.nn.DataParallel(D)


# #     model_wgan = 120
# #     # Load saved model weights
# #     G.load_state_dict(torch.load(f'checkpoints_v2/WGAN_GP_{model_wgan}_G.pth', weights_only=True))
# #     D.load_state_dict(torch.load(f'checkpoints_v2/WGAN_GP_{model_wgan}_D.pth', weights_only=True))


# #     # Parameters
# #     tau = -10000  # Adjust tau as needed
# #     num_samples = 3000
# #     batch_size = 100

# #     # Generate samples with DRS
# #     print(f'Generating {num_samples} samples with DRS (tau={tau})...')
# #     samples = generate_samples_with_DRS(G, D, num_samples, batch_size, tau)

# #     # Save samples
# #     os.makedirs('DRS_samples', exist_ok=True)
# #     for idx in range(samples.size(0)):
# #         vutils.save_image(samples[idx].view(1, 28, 28),
# #                           f'DRS_samples/{idx}.png',
# #                           normalize=True)

# import torch 
# import torchvision
# import os
# import argparse
# from tqdm import tqdm

# from model import Generator
# from utils import load_model

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate Normalizing Flow.")
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=2048,
#         help="The batch size to use for training.",
#     )
#     parser.add_argument(
#         "--plot",
#         action="store_true",
#         help="If set, randomly display generated images.",
#     )
#     args = parser.parse_args()

#     print("Model Loading...")
#     # Model Pipeline
#     mnist_dim = 784

#     model = Generator(g_output_dim = mnist_dim).cuda()
#     # model = load_model(model, 'checkpoints')
#     model = torch.nn.DataParallel(model).cuda()

#     model_wgan = 128
#     # Load saved model weights
#     model_path = f'checkpoints/WGAN_GP_{model_wgan}_G.pth'
#     model.load_state_dict(torch.load(model_path, weights_only=True))
#     model.eval()

#     print(f'Model {model_path} loaded.')

#     print("Start Generating")
#     os.makedirs("samples", exist_ok=True)

#     total_samples = 10000
#     batch_size = args.batch_size

#     with torch.no_grad():
#         for n_samples in tqdm(range(0, total_samples, batch_size)):
#             z = torch.randn(args.batch_size, 100).cuda()
#             x = model(z)
#             x = x.reshape(batch_size, 28, 28)
#             for k in range(x.shape[0]):
#                 if n_samples<10000:
#                     torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
#                     n_samples += 1

import torch 
import torchvision
import os
import argparse

from model import Generator, Discriminator_WGAN_GP
from utils import load_model, generate_samples_with_DRS
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
    tau = 4.5  # Adjust tau as needed
    num_samples = 10000
    batch_size = 100

    # Generate samples with DRS
    print(f'Generating {num_samples} samples with DRS (tau={tau})...')
    samples = generate_samples_with_DRS(G, D, num_samples, batch_size, tau)

    # Save samples
    os.makedirs('samples', exist_ok=True)
    for idx in range(samples.size(0)):
        vutils.save_image(samples[idx].view(1, 28, 28),
                          f'samples/{idx}.png',
                          normalize=True)