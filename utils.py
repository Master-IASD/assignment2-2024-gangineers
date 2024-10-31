import torch
import os
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()

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
    #=======================Train the generator=======================#
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
