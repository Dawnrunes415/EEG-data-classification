import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.signal import butter, filtfilt
# if you see: ModuleNotFoundError: No module named 'pywt'
# then "pip install PyWavelets"
import pywt
from sklearn.preprocessing import StandardScaler


########## Variables ##########
LOWCUT = 0.5
HIGHCUT = 40  
FS = 256
WINDOW_SIZE = 15
STEP_SIZE = 1
PRED_SIZE = 30

########## Loading the datasets ##########
# Load the training dataset 

path = "D:\\Vanessa\\Documents\\eeg_data\\" # Change this to the location of your local data folder

train_data = np.load(path+'eeg-predictive_train.npz')
X_train = train_data['train_signals']
y_train = train_data['train_labels']

# Load the validation dataset 
val_data = np.load(path+'eeg-predictive_val.npz')
X_val = val_data['val_signals']
y_val = val_data['val_labels']

# Load the balanced validation dataset 
# The balanced datset is a subset of the validation dataset so we will be using this for validation 
# Not balanced validation dataset will be processed and used if the balanced validation dataset is not enough
val_bal_data = np.load(path+'eeg-predictive_val_balanced.npz')
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

print("Validation labels:", np.unique(y_val, return_counts=True))
seizure_percent_val = np.sum(y_val == 1) / len(y_val) * 100
print('% Seizure in the validation set:', seizure_percent_val, '%') # 22.44%

print("Balanced Validation labels:", np.unique(y_val_bal, return_counts=True))
seizure_percent_bal_val = np.sum(y_val_bal == 1) / len(y_val_bal) * 100
print('% Seizure in the validation set:', seizure_percent_bal_val, '%')


########## Data Distribution ##########
# print("Train mean:", np.mean(X_train), "Train std:", np.std(X_train), "Train perchannel mean:", np.mean(X_train, axis=(0, 2)), "Train perchannel std:", np.std(X_train, axis=(0, 2)))
# print("Val mean:", np.mean(X_val), "Val std:", np.std(X_val), "Val perchannel mean:", np.mean(X_val, axis=(0, 2)), "Val perchannel std:", np.std(X_val, axis=(0, 2)))
# print("Val_bal mean:", np.mean(X_val_bal), "Val_bal std:", np.std(X_val_bal), "Val_bal perchannel mean:", np.mean(X_val_bal, axis=(0, 2)), "Val_bal perchannel std:", np.std(X_val_bal, axis=(0, 2)))

# import matplotlib.pyplot as plt

# for i in range(5):  # Pick a few channels
#     plt.figure()
#     plt.hist(X_train[:, i, :].flatten(), bins=100, alpha=0.5, label='Train', density=True)
#     plt.hist(X_val[:, i, :].flatten(), bins=100, alpha=0.5, label='Val', density=True)
#     plt.title(f'Channel {i}')
#     plt.legend()
#     plt.show()

# print(y_train[:50]) 
# print(y_val[:50]) 

########## Visualizing the data ##########

'''
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
'''

########## Noise removal using bandpass filter ##########
def create_bandpass_filter(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    # Normalize cutoff frequencies 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, b, a):
    # Apply the filter to each signal (each channel in each sample)
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):      
        for c in range(data.shape[1]):  
            filtered[i, c, :] = filtfilt(b, a, data[i, c, :])
    return filtered

b, a = create_bandpass_filter(lowcut=LOWCUT, highcut=HIGHCUT, fs=FS)
X_train = bandpass_filter(X_train, b, a)
X_val = bandpass_filter(X_val, b, a)
X_val_bal = bandpass_filter(X_val_bal, b, a)

########## Artifacts removal ##########

# This function uses PyWavelets to remove artifacts.
# Level - Range: [1, max_level], Meaning: differentiate approximation and detail of wavelets
#         Usage: higher level, higher computation, finer each bandwidth (Hz)
# Mode - Range: (soft, garrote, hard, greater, less), Meaning: defines thresholding mode
#        Usage: remove any potential noises after layering the signals into wavelets;
#        "soft" thresholding in this place does: "data values with absolute value 
#        less than param are replaced with substitute. Data values with absolute value greater 
#        or equal to the thresholding value are shrunk toward zero by value. (PyWavelets docs)"
def wavelet_denoise(data, level=None, mode='soft'):
    # db4 being the best time/frequency cutoff for EEG signals
    wavelet = "db4"

    if np.std(data) < 1e-4:
        return data.copy()
    
    if level is None:
        # will use the maximal level if level is None (default)
        level = pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len)
    
    # Multilevel decomposition platform (pywt.wavedec)
    # receiving wavelet coefficients from wavedec
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # bro tbh I don't comprehend exactly what wavelet coefficiencts are for
    # but my understanding is that through some convolutional (or whatever) calculations
    # you will get a segmented analysis of the wavelets.

    # Median Absolute Deviation method:
    # using the highest layer of coeffs (represent high frequency, usually noises)
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    # Remove coeffs of what likely to be random noises

    if np.isclose(sigma, 0):
        return data.copy()

    universal_threshold = sigma * np.sqrt(2 * np.log(len(data)))

    # coeffs[0] are the ones that describe the most general pattern
    coeffs_thresh = [coeffs[0]]

    for i in range(1, len(coeffs)):
        coeffs_thresh.append(pywt.threshold(coeffs[i], value=universal_threshold, mode=mode))
    
    # Multileverl reconstruct EEG signals
    return pywt.waverec(coeffs_thresh, wavelet)

# This is the actual function to be called on to remove artifacts.
# It will apply wavelet denoising to each channel of each sample. 
def denoise(dataset):
    denoised_dataset = np.zeros_like(dataset)
    for s in range (dataset.shape[0]):
        for c in range (dataset.shape[1]):
            # denoised_channel = wavelet_denoise(channel)
            # denoised_dataset.append(denoised_channel)
            denoised_dataset[s, c] = wavelet_denoise(dataset[s, c])
    return denoised_dataset

# Plotting the first 5 graphs for visualization.
ch_names = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3','P3-O1',
             'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
             'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8']

# for i in range(5):
#     sample_signal = X_train[i]
#     denoised = denoise(sample_signal)

#     # These should give 23 and 256 
#     channels = sample_signal.shape[0]
#     time_points = sample_signal.shape[1]

#     eeg, time = plt.subplots(channels, 1, figsize=(10, 20), sharex=True)

#     # Assign names for the channels 1-23
#     for c in range(channels):
#         time[c].plot(sample_signal[c], label="Raw")
#         time[c].plot(denoised[c], label="Denoised")
#         time[c].set_ylabel(ch_names[c])
#         time[c].legend(loc='upper right')
#     time[-1].set_xlabel("Time: 256 points")

#     # Convert the labels to string
#     if int(y_train[i]) == 1:
#         label = "Seizure"
#     else: 
#         label = "Not Seizure"
#     eeg.suptitle(f"EEG Sample {i+1} - Label: {label}")
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     # Save the figure into the eeg_graphs folder
#     plt.savefig(f"eeg_denoised_graphs/sample_{i+1}.png")
#     plt.close()

########## Normalization ##########

def get_scalers(X_train):
    scalers = {} # Storing scalers for each channel 
    for c in range(X_train.shape[1]):
        # Reshape to (samples * timesteps, 1)
        # Fit on trainign data then apply to validation data
        scaler = StandardScaler()
        X_train_2d = X_train[:, c, :].reshape(-1, 1)
        scaler.fit(X_train_2d)
        scalers[c] = scaler 

    return scalers
    
def normalization(X, scalers):
    # # 2D flatten the training data to (8282, 23*256)
    # # 2D flatten the validation data to (1462, 23*256)
    # X_train_2d = X_train.reshape(X_train.shape[0], -1)
    # X_val_bal_2d = X_val_bal.reshape(X_val_bal.shape[0], -1)

    # # Standardize to mean = 0, standard deviation = 1
    # # Only using transform yet not fit for validation data because
    # # don't want model to know the fitting parameters!
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train_2d)
    # X_val_bal_scaled = scaler.transform(X_val_bal_2d)

    # # Reshape training data from (8282, 23*256) to (8282, 23, 256)
    # # Reshape validation data from (1462, 23*256) to (1462, 23, 256)
    # X_train_final = X_train_scaled.reshape(X_train.shape)
    # X_val_bal_final = X_val_bal_scaled.reshape(X_val_bal.shape)
    # return X_train_final, X_val_bal_final

    # Channel-wise normalization. 
    # This maintains amplitude differences between channels. 
    # Flattening everything destroys temporal information as it treats 
    # each channel as the same. 
    X_final = np.zeros_like(X)
    for c in range(X.shape[1]):
        X_2d = X[:, c, :].reshape(-1, 1)
        # Transform data then reshape it back to per-sample format --> (Sample, 256)
        X_scaled = scalers[c].transform(X_2d)
        X_final[:, c, :] = X_scaled.reshape(X.shape[0], X.shape[2])
    return X_final 

########## Create Sliding Windows ##########

def create_sliding_window(X, y, window_size, step_size, prediction_size):
    """
    Creates sliding windows for data 
    Parameters:
        X: shape (N, 23, 256) 1-second samples
        y: shape (N,) labels per sample (0/1)
        window_size: seconds of EEG used as input
        step_size: size to slide the window (seconds)
        prediction_size: seconds into the future to check for seizure
    """

    X_windows = []
    y_windows = []

    total_samples = X.shape[0]

    for i in range(0, total_samples - window_size - prediction_size + 1, step_size):
        window = X[i : i + window_size] # Frome 0 to window_size
        future_labels = y[i + window_size : i + window_size + prediction_size] # From end of window to end of prediction size
        seizure_fraction = np.sum(future_labels) / prediction_size
        label = 1 if seizure_fraction >= 0.3 else 0 

        # If there are at least 5 consecutive 1s in the future labels, label it as seizure
        # This is to avoid false seizure detection due to short spikes in the data
        # count = 0
        # label = 0
        # for val in future_labels:
        #     if val == 1:
        #         count += 1
        #         if count >= 3:
        #             label = 1
        #             break
        #     else:
        #         count = 0

        X_windows.append(window)
        y_windows.append(label)
    
    return np.array(X_windows), np.array(y_windows)

########## Saving Data ##########

scalers = get_scalers(X_train)

# This is the denoised + normalized one
X_train_denoised = denoise(X_train)
X_val_denoised = denoise(X_val)
X_train_norm_denoise, X_val_norm_denoise = normalization(X_train_denoised, scalers), normalization(X_val_denoised, scalers)
X_train_denoise_win, y_train_win = create_sliding_window(X_train_norm_denoise, y_train, WINDOW_SIZE, STEP_SIZE, PRED_SIZE)
X_val_denoise_win, y_val_win = create_sliding_window(X_val_norm_denoise, y_val, WINDOW_SIZE, STEP_SIZE, PRED_SIZE)
np.savez_compressed(
    'denoised_eeg_data.npz',
    X_train=X_train_denoise_win,
    X_val=X_val_denoise_win,
    y_train=y_train_win,
    y_val=y_val_win
)

# np.savez_compressed(
#     'denoised_eeg_data.npz',
#     X_train=X_train_norm_denoise,
#     X_val=X_val_norm_denoise,
#     y_train=y_train,
#     y_val=y_val_bal
# )

# This is the ONLY normalized one
X_train_norm, X_val_norm = normalization(X_train, scalers), normalization(X_val, scalers)
X_train_win, y_train_win = create_sliding_window(X_train_norm, y_train, WINDOW_SIZE, STEP_SIZE, PRED_SIZE)
X_val_win, y_val_win = create_sliding_window(X_val_norm, y_val, WINDOW_SIZE, STEP_SIZE, PRED_SIZE)
np.savez_compressed(
    'processed_eeg_data.npz',
    X_train=X_train_win,
    X_val=X_val_win,
    y_train=y_train_win,
    y_val=y_val_win
)

# np.savez_compressed(
#     'processed_eeg_data.npz',
#     X_train=X_train_norm,
#     X_val=X_val_norm,
#     y_train=y_train,
#     y_val=y_val_bal
# )

