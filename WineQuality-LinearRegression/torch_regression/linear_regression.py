import time
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_funcs import train_model, validate_model
from model import MLP
from data.custom_dataset import CustomDataset

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sizes = [11, 86, 1]
    lr = 0.01588
    weight_decay = 0.33334
    batch_size = 8
    num_epochs = 200

    mlp = MLP(sizes)
    mlp.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    trained_models_dir = f'trained_models/model_{sizes[1]}_{lr}_{weight_decay}'

    train_dataset = CustomDataset('../data/wine+quality/train-red.csv')
    val_dataset = CustomDataset('../data/wine+quality/val-red.csv')
    test_dataset = CustomDataset('../data/wine+quality/test-red.csv')

    train_loader = DataLoader(ConcatDataset([train_dataset, val_dataset]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size)

    train_writer = SummaryWriter(f'{trained_models_dir}/logs/train')
    val_writer = SummaryWriter(f'{trained_models_dir}/logs/val')

    best_loss = np.inf

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        time.sleep(0.1)

        train_loss = train_model(mlp, train_loader, criterion, optimizer, device)

        train_writer.add_scalar('loss', train_loss, epoch + 1)

        time.sleep(0.1)
        print(f'loss: {train_loss:.3f}')
        time.sleep(0.1)

        val_loss = validate_model(mlp, val_loader, criterion, device)

        val_writer.add_scalar('loss', val_loss, epoch + 1)

        time.sleep(0.1)
        print(f'loss: {val_loss:.3f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(mlp.state_dict(), f'{trained_models_dir}/best_model_{epoch + 1}.pth')

        time.sleep(0.1)

    train_writer.close()
    val_writer.close()
