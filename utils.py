import torch
import os
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1).cuda()
    alpha = alpha.expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = interpolates.requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1).cuda()
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def D_train_WGAN_GP(x, G, D, D_optimizer, lambda_gp):
    D.zero_grad()

    # Real data
    x_real = x.cuda()
    T_real = D(x_real)
    D_real_loss = -torch.mean(T_real)

    # Fake data
    z = torch.randn(x.size(0), 100).cuda()
    x_fake = G(z).detach()
    T_fake = D(x_fake)
    D_fake_loss = torch.mean(T_fake)

    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(D, x_real, x_fake)

    # Total loss
    D_loss = D_real_loss + D_fake_loss + lambda_gp * gradient_penalty
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item(), T_real.mean().item(), T_fake.mean().item()

def G_train_WGAN_GP(G, D, G_optimizer, batch_size):
    G.zero_grad()

    z = torch.randn(batch_size, 100).cuda()
    x_fake = G(z)
    T_fake = D(x_fake)
    G_loss = -torch.mean(T_fake)

    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()

def D_train(x, G, D, D_optimizer, criterion):
    # =======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

    D_output = D(x_fake)

    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()

def D_train_KL(x, G, D, D_optimizer):
    D.zero_grad()

    # Real data
    x_real = x.cuda()
    T_real = D(x_real)
    D_real_loss = -torch.mean(T_real)

    # Fake data
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake = G(z).detach()
    T_fake = D(x_fake)
    D_fake_loss = torch.mean(torch.exp(T_fake - 1))

    # Total loss
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()



def G_train(x, G, D, G_optimizer, criterion):
    # =======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    y = torch.ones(x.shape[0], 1).cuda()

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

def G_train_KL(x, G, D, G_optimizer):
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    x_fake = G(z)
    T_fake = D(x_fake)
    G_loss = torch.mean(torch.exp(T_fake - 1))

    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()



def save_models(G, D, folder, prefix=''):
    torch.save(G.state_dict(), os.path.join(folder, prefix+'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder, prefix+'D.pth'))


def load_model(G, folder, prefix=''):
    ckpt = torch.load(os.path.join(folder, prefix+'G.pth'))
    print(f"Loading model from {os.path.join(folder, prefix+'G.pth')}") 
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def generate_samples_with_DRS(G, D, num_samples, batch_size, tau):
    G.eval()
    D.eval()
    samples = []
    total_generated = 0
    total_attempted = 0

    while total_generated < num_samples:
        # Generate latent vectors
        z = torch.randn(batch_size, 100).cuda()
        with torch.no_grad():
            # Generate samples
            x_fake = G(z)
            # Compute discriminator logits
            D_output = D(x_fake).squeeze()
            # Compute acceptance probabilities
            acceptance_probs = torch.sigmoid(D_output - tau)
            # Sample from Bernoulli distribution
            accept = torch.bernoulli(acceptance_probs).bool()
            # Select accepted samples
            accepted_samples = x_fake[accept]
            samples.append(accepted_samples.cpu())
            total_generated += accepted_samples.size(0)
            total_attempted += batch_size
        
            # if total_generated != 0  and total_generated % 100 == 0:
            #     print(f'Generated {total_generated}/{num_samples} samples')

    acceptance_rate = total_generated / total_attempted
    print(f'Acceptance Rate: {acceptance_rate:.4f}')

    G.train()
    D.train()

    # Concatenate all accepted samples
    samples = torch.cat(samples, dim=0)
    # If more samples than needed, truncate
    samples = samples[:num_samples]

    return samples