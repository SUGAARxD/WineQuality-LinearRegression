import torch
from tqdm import tqdm


def train_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in tqdm(loader, desc='Train', colour='magenta'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # batch loss
        loss = criterion(outputs, labels)
        # batch weighted loss
        total_loss += loss.item() * labels.size(0)

        # calculate and update gradients
        loss.backward()
        optimizer.step()

    # entire epoch loss
    total_loss /= len(loader.dataset)
    return total_loss


def validate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation', colour='magenta'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # batch loss
            loss = criterion(outputs, labels)
            # batch weighted loss
            total_loss += loss.item() * labels.size(0)

    # entire epoch loss
    total_loss /= len(loader.dataset)
    return total_loss
