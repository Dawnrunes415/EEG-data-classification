import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.signal import butter, filtfilt

########## Loading the datasets ##########
# Load the training dataset 
train_data = np.load('data/eeg-predictive_train.npz')
print(train_data.files) # Check the keys in the npz file
X_train = train_data['train_signals']
y_train = train_data['train_labels']

# Load the validation dataset 
val_data = np.load('data/eeg-predictive_val.npz')
X_val = val_data['val_signals']
y_val = val_data['val_labels']

# Load the balanced validation dataset 
# The balanced datset is a subset of the validation dataset so we will be using this for validation 
# Not balanced validation dataset will be processed and used if the balanced validation dataset is not enough
val_bal_data = np.load('data/eeg-predictive_val_balanced.npz')
X_val_bal = val_bal_data['val_signals']
y_val_bal = val_bal_data['val_labels']

########## Printing the Shape and Distribution ##########
print("X_train:", X_train.shape, " y_train:", y_train.shape) # (8282, 23, 256) 
print("X_val:", X_val.shape, " y_val:", y_val.shape) # (1462, 23, 256)
print("X_val_bal:", X_val_bal.shape, " y_val_bal:", y_val_bal.shape) # (656, 23, 256)

print("Train label counts:", np.bincount(y_train.astype(int)))
print("Val label counts:", np.bincount(y_val.astype(int)))
print("Balanced label counts:", np.bincount(y_val_bal.astype(int)))

print("Train labels:", np.unique(y_train, return_counts=True))
seizure_percent_train = np.sum(y_train == 1) / len(y_train) * 100
print('% Seizure in the training set:', seizure_percent_train, '%') # 21.67%

print("Balanced validation labels:", np.unique(y_val, return_counts=True))
seizure_percent_val = np.sum(y_val == 1) / len(y_val) * 100
print('% Seizure in the bal.validation set:', seizure_percent_val, '%') # 22.44%

########## Visualizing the data ##########
# Visualize the first five samples in the training set 
for i in range(5):
    sample_signal = X_train[i]

    # These should give 23 and 256 
    channels = sample_signal.shape[0]
    time_points = sample_signal.shape[1]

    eeg, time = plt.subplots(channels, 1, figsize=(10, 20), sharex=True)

    # Assign names for the channels 1-23
    for c in range(channels):
        time[c].plot(sample_signal[c])
        time[c].set_ylabel(f"Ch {c+1}") 

    time[-1].set_xlabel("Time: 256 points")
    # Convert the labels to string
    if int(y_train[i]) == 1:
        label = "Seizure"
    else: 
        label = "Not Seizure"
    eeg.suptitle(f"EEG Sample {i+1} - Label: {label}")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # Save the figure into the eeg_graphs folder
    plt.savefig(f"eeg_graphs/sample_{i+1}.png")
    #plt.show()
    plt.close()


########## Noise removal using bandpass filter ##########
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    # Normalize cutoff frequencies 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter to each signal (each channel in each sample)
    filtered = np.empty_like(data)
    for i in range(data.shape[0]):      
        for c in range(data.shape[1]):  
            filtered[i, c, :] = filtfilt(b, a, data[i, c, :])
    return filtered

X_train = bandpass_filter(X_train, lowcut=0.5, highcut=40, fs=256)
X_val = bandpass_filter(X_val, lowcut=0.5, highcut=40, fs=256)
X_val_bal = bandpass_filter(X_val_bal, lowcut=0.5, highcut=40, fs=256)


########## Artifacts removal ##########




########## Downsampling ##########
# I move this part after artifacts and noise removal because it makes sense to clean the data first 





########## Normalization ##########








