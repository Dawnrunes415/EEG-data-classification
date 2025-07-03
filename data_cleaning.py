import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.signal import butter, filtfilt
# if you see: ModuleNotFoundError: No module named 'pywt'
# then "pip install PyWavelets"
import pywt
from sklearn.preprocessing import StandardScaler

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

'''

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

'''

########## Artifacts removal ##########

# This is a helper function to avoid invalid true-divide of soft thresholding.
# I do not like ValueError, this really shouldn't be here, very unhealthy, but let it be now.
def safe_soft_threshold(coeff):
    m_coeff = []
    for c in coeff:
        if c == 0:
            m_coeff.append(1e-8)
        else:
            m_coeff.append(c)
    return m_coeff

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
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    # Remove coeffs of what likely to be random noises
    universal_threshold = sigma * np.sqrt(2 * np.log(len(data)))

    # coeffs[0] are the ones that describe the most general pattern
    coeffs_thresh = [coeffs[0]]
    for i in range(1, len(coeffs)):
        # perform soft thresholding
        coeffs_thresh.append(pywt.threshold(safe_soft_threshold(coeffs[i]), value=universal_threshold, mode=mode))
    
    # Multileverl reconstruct EEG signals
    return pywt.waverec(coeffs_thresh, wavelet)

# This is the actual function to be called on to remove artifacts.
def denoise(dataset):
    denoised_dataset = []
    for channel in dataset:
        denoised_channel = wavelet_denoise(channel)
        denoised_dataset.append(denoised_channel)
    return np.array(denoised_dataset)

# Plotting the first 5 graphs for visualization.
for i in range(5):
    sample_signal = X_train[i]
    denoised = denoise(sample_signal)

    # These should give 23 and 256 
    channels = sample_signal.shape[0]
    time_points = sample_signal.shape[1]

    eeg, time = plt.subplots(channels, 1, figsize=(10, 20), sharex=True)

    # Assign names for the channels 1-23
    for c in range(channels):
        time[c].plot(sample_signal[c], label="Raw")
        time[c].plot(denoised[c], label="Denoised")
        time[c].set_ylabel(f"Ch {c+1}")
        time[c].legend(loc='upper right')
    time[-1].set_xlabel("Time: 256 points")

    # Convert the labels to string
    if int(y_train[i]) == 1:
        label = "Seizure"
    else: 
        label = "Not Seizure"
    eeg.suptitle(f"EEG Sample {i+1} - Label: {label}")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # Save the figure into the eeg_graphs folder
    plt.savefig(f"eeg_denoised_graphs/sample_{i+1}.png")
    plt.close()

########## Upweighting ##########
# I move this part after artifacts and noise removal because it makes sense to clean the data first 

# Changed from downsampling -> upweighting as it avoids the challenges of insufficient data;
# yet ensured balanced data.

########## Normalization ##########

# 2D flatten the training data to (8282, 23*256)
# 2D flatten the validation data to (1462, 23*256)
X_train_2d = X_train.reshape(X_train.shape[0], -1)
X_val_bal_2d = X_val_bal.reshape(X_val_bal.shape[0], -1)

# Standardize to mean = 0, standard deviation = 1
# Only using transform yet not fit for validation data because
# don't want model to know the fitting parameters!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_2d)
X_val_bal_scaled = scaler.transform(X_val_bal_2d)

# Reshape training data from (8282, 23*256) to (8282, 23, 256)
# Reshape validation data from (1462, 23*256) to (1462, 23, 256)
X_train_final = X_train_scaled.reshape(X_train.shape)
X_val_bal_final = X_val_bal_scaled.reshape(X_val_bal.shape)

########## Saving Data ##########

np.savez_compressed(
    'processed_eeg_data.npz',
    X_train=X_train_final,
    X_val=X_val_bal_final,
    y_train=y_train,
    y_val=y_val_bal
)