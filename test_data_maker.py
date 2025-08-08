import numpy as np
import torch


path = "C:\\Users\\andre\\OneDrive\\Documents\\data\\" # Change this to the location of your local data folder

# Loading datasets
test_superset = np.load(path+'eeg-predictive_val.npz')
x_superset = torch.from_numpy(test_superset['val_signals'])
y_superset = torch.from_numpy(test_superset['val_labels'])


# This will be the test set, eliminating these samples from the superset
test_set = np.load(path+'eeg-predictive_val_balanced.npz')
x_test = torch.from_numpy(test_set['val_signals'])
y_test = torch.from_numpy(test_set['val_labels'])

print(x_superset.shape, x_test.shape)

print(sum(y_superset), sum(y_test))

same = []
for i in range(len(x_superset)):
    for j in range(len(x_test)):
        if torch.equal(x_superset[i], x_test[j]):
            same.append(i)

remaining = [i for i in range(len(x_superset)) if i not in same]

remaining_x = x_superset[remaining].numpy()
remaining_y = y_superset[remaining].numpy()

np.savez(path+'eeg-predictive_val_remaining.npz', val_signals=remaining_x, val_labels=remaining_y)
