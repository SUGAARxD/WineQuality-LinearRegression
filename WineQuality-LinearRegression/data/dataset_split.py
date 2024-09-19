import numpy as np
import pandas as pd

dataset = pd.read_csv('wine+quality/winequality-red.csv').values

# if there are missing values in the dataset, those missing spots will be filled with the mean of existent values
columns_mean = np.round(np.nanmean(dataset, axis=0), decimals=5)

dataset = np.where(np.isnan(dataset), columns_mean, dataset)

np.random.shuffle(dataset)

train_val_ratio = 0.65
train_ratio = 0.7

train_val_dataset = dataset[:int(train_val_ratio * len(dataset))]
test_dataset = dataset[int(train_val_ratio * len(dataset)):]
train_dataset = train_val_dataset[:int(train_ratio * len(train_val_dataset))]
val_dataset = train_val_dataset[int(train_ratio * len(train_val_dataset)):]

train_df = pd.DataFrame(train_dataset)
val_df = pd.DataFrame(val_dataset)
test_df = pd.DataFrame(test_dataset)

train_df.to_csv('wine+quality/train-red.csv', index=False, header=False)
val_df.to_csv('wine+quality/val-red.csv', index=False, header=False)
test_df.to_csv('wine+quality/test-red.csv', index=False, header=False)
