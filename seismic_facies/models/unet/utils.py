import torch
import torch.nn as nn
from tqdm import tqdm
from livelossplot import PlotLosses

from seismic_facies.models.loss import dice_loss_weighted

def train(model, optimizer, ce_loss, alpha, data_loader, device):
    model.train()
    train_loss = 0.
    for x, y, _ in data_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = ce_loss(logits, y) + alpha * dice_loss_weighted(logits, y)# + v2 * focal_loss_from_logits(logits, y)#, alpha=class_weights.to(device))
        loss.backward()
        train_loss += loss*x.size(0)
        optimizer.step()
    train_loss = train_loss / len(data_loader.dataset)
    return train_loss


def valid(model, ce_loss, alpha, data_loader, device):
    model.eval()
    valid_loss = 0.
    with torch.no_grad():
        for x, y, _ in data_loader:
          x, y = x.to(device), y.to(device)
          logits = model(x)
          loss = ce_loss(logits, y) + alpha * dice_loss_weighted(logits, y)# + v2 * focal_loss_from_logits(logits, y)#, alpha=class_weights.to(device))
          valid_loss += loss*x.size(0)
        valid_loss = valid_loss / len(data_loader.dataset)
        return valid_loss

def training_loop(model, optimizer, class_weights, alpha, train_loader, valid_loader, device, nepochs):
    ce_loss = nn.CrossEntropyLoss(weight = class_weights.to(device) if class_weights is not None else None)
    liveloss = PlotLosses()
    for i in tqdm(range(nepochs)):
        train_loss = train(model, optimizer, ce_loss, alpha, train_loader, device)
        valid_loss = valid(model, ce_loss, alpha, valid_loader, device)
        # Liveloss plot
        logs = {}
        logs['' + 'log loss'] = train_loss.item()
        logs['val_' + 'log loss'] = valid_loss.item()
        liveloss.update(logs)
        liveloss.draw()