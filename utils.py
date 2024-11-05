import torch
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")


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