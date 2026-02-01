import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from livelossplot import PlotLosses

from seismic_facies.models.loss import dice_loss_weighted

def get_diff_model_params(dim=64, T=1000):
    dim = dim
    T = T
    t = torch.linspace(1, T, T + 1)
    s = 0.0081

    y = torch.cos((((t / T) + s) / (1 + s)) * torch.pi / 2) ** 2

    alphas_bar = y / y[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = torch.clip(betas, 1e-8, 0.999)
    alphas = 1 -  betas
    alphas_bar = torch.cumprod(alphas, axis=0) 

    return dim, alphas, alphas_bar, betas, T

# define a function to handle the discrepancy between batch vs schedule shapes
def fix_batch(sched, t, device):
    """extract sched at time t, and expand to batch dimensionality"""
    return sched.to(device)[t.to(device)[:,None,None,None]]

# let's define a function that implements the forward diffusion process
def forward_diffusion(x0, t, e, alphas_bar):
    """run the forward diffusion process on x up to time t using noise e"""
    sqrt_alpha_term = fix_batch(torch.sqrt(alphas_bar), t, x0.device) * x0
    one_minus_sqrt_term = fix_batch(torch.sqrt(1 - alphas_bar), t, x0.device) * e
    return sqrt_alpha_term + one_minus_sqrt_term

# let's define a function that implements the reverse diffusion process
def reverse_diffusion(x, t, e, z, betas, alphas, alphas_bar):
    """run the reverse diffusion process"""
    sigma = fix_batch(torch.sqrt(betas), t, x.device)
    sqrt_recip_alphas = fix_batch(torch.sqrt(1.0 / alphas), t, x.device)
    scale = fix_batch((1 - alphas) / torch.sqrt(1. - alphas_bar), t, x.device)

    return sqrt_recip_alphas * (x - scale*e) + sigma*z

# define the sinusoidal position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    """sinusoidal position embedding, https://arxiv.org/abs/1706.03762"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
def train(model, optimizer, mse_loss, ce_loss, alp, bt, train_loader, device, T, alphas, alphas_bar, betas):
    model.train()
    pbar = tqdm(train_loader)
    train_tot_loss = 0.0

    for X, y, _ in pbar:
        X = X.to(device)
        if y.ndim == 4:
          y_idx = y.squeeze(1).to(device).long()
        else:
          y_idx = y.to(device).long()


        #one-hot labels
        y0 = torch.nn.functional.one_hot(y_idx, num_classes=6).permute(0,3,1,2).float()

        # zero the gradients
        optimizer.zero_grad()

        # sample t from a uniform distribution
        t = torch.randint(0, T, (y0.shape[0], ), dtype=torch.long).to(device)

        # sample e from a normal distribution
        e = torch.randn_like(y0)

        # run the forward diffusion process to add noise to x0
        y_t = forward_diffusion(y0, t, e, alphas_bar)

        # calculate the loss between the predicted noise and the true noise
        e_pred = model(y_t, t, X)

        loss_eps = mse_loss(e_pred, e)

        pred = torch.randn_like(y0)

        for i in reversed(range(0,T)):

          # sample z from a normal distribution if condition met
          if i > 0:
              z = torch.randn_like(pred)
          else:
              z = 0

          # convert t to a tensor, send to the GPU, and expand its first dimension to be equal to batch size
          i = torch.tensor(i).to(device).expand(pred.shape[0])

          # denoise x
          pred = reverse_diffusion(pred, i, e_pred, z, betas, alphas, alphas_bar)

        # ONLY apply CE when noise is low
        mask = (t < T)  

        if mask.any():
            loss_seg = ce_loss(pred[mask], y_idx[mask]) + bt * dice_loss_weighted(pred[mask], y_idx[mask])
        else:
            loss_seg = 0.0

        loss = loss_eps + alp * loss_seg

        # backpropagation to obtain gradients w.r.t model parameters
        loss.backward()

        # take an optimisation step
        optimizer.step()

        # Report current loss using tqdm
        train_tot_loss += loss.item()
        pbar.set_description(f"train_loss: {loss.item():.4f}")

    # average training loss for the epoch (float)
    train_loss = train_tot_loss / len(train_loader)
    return train_loss

def valid(model, mse_loss, ce_loss, alp, bt, val_loader, device, T, alphas, alphas_bar, betas):
    model.eval()
    val_tot_loss = 0.0

    with torch.no_grad():
        val_pbar = tqdm(val_loader)
        for X, y, _ in val_pbar:
            X = X.to(device)
            if y.ndim == 4:
              y_idx = y.squeeze(1).to(device).long()
            else:
              y_idx = y.to(device).long()
            #y0 = y_idx.unsqueeze(1).float()  # (B,1,H,W)
            y0 = torch.nn.functional.one_hot(y_idx, num_classes=6).permute(0,3,1,2).float()
            #y0 = y0 * 2 - 1

            # sample t from a uniform distribution
            t = torch.randint(0, T, (y0.shape[0], ), dtype=torch.long).to(device)

            # sample e from a normal distribution
            e = torch.randn_like(y0)

            # forward diffusion
            y_t = forward_diffusion(y0, t, e, alphas_bar)

            # predict noise
            e_pred = model(y_t, t, X)

            loss_eps = mse_loss(e_pred, e)

            pred = torch.randn_like(y0)

            for i in reversed(range(0,T)):

              # sample z from a normal distribution if condition met
              if i > 0:
                  z = torch.randn_like(pred)
              else:
                  z = 0

              # convert t to a tensor, send to the GPU, and expand its first dimension to be equal to batch size
              i = torch.tensor(i).to(device).expand(pred.shape[0])

              # denoise x
              pred = reverse_diffusion(pred, i, e_pred, z, betas, alphas, alphas_bar)

            # ONLY apply CE when noise is low
            mask = (t < T)   

            if mask.any():
                loss_seg = ce_loss(pred[mask], y_idx[mask]) + bt * dice_loss_weighted(pred[mask], y_idx[mask])
            else:
                loss_seg = 0.0

            val_loss_batch = loss_eps + alp * loss_seg

            val_tot_loss += val_loss_batch.item()
            val_pbar.set_description(f"val_loss: {val_loss_batch.item():.4f}")

    # average validation loss for the epoch (float)
    val_loss = val_tot_loss / len(val_loader)
    return val_loss

def training_loop(model, optimizer, class_weights, alp, bt, train_loader, val_loader, device, nepochs, T, alphas, alphas_bar, betas):
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(weight = class_weights.to(device) if class_weights is not None else None)
    liveloss = PlotLosses(groups={'loss': ['train_loss', 'val_loss']})

    for epoch in range(nepochs):
        print(f"Epoch {epoch+1}/{nepochs}")

        train_loss = train(model, optimizer, mse_loss, ce_loss, alp, bt, train_loader, device, T, alphas, alphas_bar, betas)
        print(f"Training Loss: {train_loss:.4f}")

        val_loss = valid(model, mse_loss, ce_loss, alp, bt, val_loader, device, T, alphas, alphas_bar, betas)
        print(f"Validation Loss: {val_loss:.4f}")
        
        logs = {
            'train_loss': float(train_loss),  
            'val_loss': float(val_loss),
        }
        liveloss.update(logs)
        liveloss.draw()