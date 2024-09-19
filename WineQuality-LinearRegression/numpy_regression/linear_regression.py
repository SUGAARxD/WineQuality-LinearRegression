import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from model_funcs import normalize_dataset, train_model, validate_model
from model import MLP


train_dataset = pd.read_csv('../data/wine+quality/train-red.csv', header=None).values
val_dataset = pd.read_csv('../data/wine+quality/val-red.csv', header=None).values
test_dataset = pd.read_csv('../data/wine+quality/test-red.csv', header=None).values

train_features, train_targets = train_dataset[:, :-1], train_dataset[:, -1]
test_features, test_targets = test_dataset[:, :-1], test_dataset[:, -1]
val_features, val_targets = val_dataset[:, :-1], val_dataset[:, -1]

train_val_features = np.concatenate((train_features, val_features), axis=0)
train_val_targets = np.concatenate((train_targets, val_targets), axis=0)

normalize_dataset(train_features)
normalize_dataset(test_features)
normalize_dataset(val_features)

normalize_dataset(train_val_features)

sizes = [11, 20, 1]
lr = 1e-1
weight_decay = 1e-2
batch_size = 8
num_epochs = 200

scale = np.sqrt(2 / sizes[0])

mlp = MLP(sizes, scale, lr, weight_decay)

trained_models_dir = f'trained_models/model_{sizes[1]}_{lr}_{weight_decay}'

train_writer = SummaryWriter(f'{trained_models_dir}/logs/train')
val_writer = SummaryWriter(f'{trained_models_dir}/logs/val')

best_loss = np.inf

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch + 1}/{num_epochs}')

    train_idx_perm = np.array_split(np.random.permutation(train_val_features.shape[0]),
                                    int(np.ceil(train_val_features.shape[0] / batch_size)))
    val_idx_perm = np.array_split(np.random.permutation(test_features.shape[0]),
                                  int(np.ceil(test_features.shape[0] / batch_size)))

    train_features_perm = [train_val_features[idx] for idx in train_idx_perm]
    train_targets_perm = [train_val_targets[idx] for idx in train_idx_perm]

    val_features_perm = [test_features[idx] for idx in val_idx_perm]
    val_targets_perm = [test_targets[idx] for idx in val_idx_perm]

    train_loss = train_model(mlp, train_features_perm, train_targets_perm)

    train_writer.add_scalar('loss', train_loss, epoch + 1)

    print(f'Train: \nloss: {train_loss:.3f}')

    val_loss = validate_model(mlp, val_features_perm, val_targets_perm)

    val_writer.add_scalar('loss', val_loss, epoch + 1)

    print(f'Validation: \nloss: {val_loss:.3f}')

    if val_loss < best_loss:
        best_loss = val_loss
        np.save(f'{trained_models_dir}/best_model_{epoch + 1}.npy', mlp.weights)

train_writer.close()
val_writer.close()
