import numpy as np


def normalize_dataset(dataset):
    mean = np.mean(dataset, axis=0)
    std_dev = np.std(dataset, axis=0)
    dataset -= mean
    dataset /= (std_dev + 1e-10)


def Leaky_ReLU(z):
    return np.maximum(0.01 * z, z)


def Leaky_ReLU_derivative(z):
    return np.where(z > 0, 1, 0.01)


def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def MSE_derivative(y_true, y_pred):
    return -2/y_true.shape[0] * (y_true - y_pred)


def train_model(model, batch_features, batch_targets):
    total_loss = 0.0
    number_of_data = 0
    for features, targets in zip(batch_features, batch_targets):
        predicted_targets = model.forward(features)

        # batch loss
        loss = MSE(targets.reshape(-1, 1), predicted_targets)
        # batch weighted loss
        total_loss += loss * targets.shape[0]

        number_of_data += targets.shape[0]

        # calculate and update gradients
        model.backward(targets.reshape(-1, 1))

    # entire epoch loss
    total_loss /= number_of_data
    return total_loss


def validate_model(model, batch_features, batch_targets):
    total_loss = 0.0
    number_of_data = 0
    for features, targets in zip(batch_features, batch_targets):
        predicted_targets = model.forward(features)

        # batch loss
        loss = MSE(targets.reshape(-1, 1), predicted_targets)
        # batch weighted loss
        total_loss += loss * targets.shape[0]

        number_of_data += targets.shape[0]

    # entire epoch loss
    total_loss /= number_of_data
    return total_loss
