import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt

def lr_finder(model, dataloader, optimizer, criterion, device, start_lr=1e-8, end_lr=10, num_iters=100):
    model.train()
    lrs, losses = [], []

    # Exponential LR schedule
    lr_lambda = lambda x: (end_lr / start_lr) ** (x / num_iters)
    lr = start_lr
    optimizer.param_groups[0]['lr'] = lr

    best_loss = float('inf')
    avg_loss = 0.0
    beta = 0.98  # smoothing

    for batch_num, (inputs, targets) in enumerate(dataloader):
        if batch_num > num_iters:
            break

        # Compute smoothed LR
        lr = start_lr * (end_lr / start_lr) ** (batch_num / num_iters)
        optimizer.param_groups[0]['lr'] = lr

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Smooth the loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (batch_num + 1))

        # Record the values
        lrs.append(lr)
        losses.append(smoothed_loss)

        # Stop if the loss explodes
        if batch_num > 10 and smoothed_loss > 4 * best_loss:
            break

        if smoothed_loss < best_loss or batch_num == 0:
            best_loss = smoothed_loss

        loss.backward()
        optimizer.step()

    #plt.plot(lrs, losses)
    #plt.xscale('log')
    #plt.xlabel('Learning Rate')
    #plt.ylabel('Loss')
    #plt.title('Learning Rate Finder')
    #plt.show()
    for i in range(len(lrs)):
        print(lrs[i], losses[i])

    exit()
