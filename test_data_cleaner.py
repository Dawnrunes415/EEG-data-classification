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
# Load the test dataset 

path = "C:\\Users\\andre\\OneDrive\\Documents\\data\\" # Change this to the location of your local data folder

test_data = np.load(path+'eeg-predictive_val_balanced.npz')
x_test = test_data['val_signals']
y_test = test_data['val_labels']

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
x_test = bandpass_filter(x_test, b, a)


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


def denoise(dataset):
    denoised_dataset = np.zeros_like(dataset)
    for s in range (dataset.shape[0]):
        for c in range (dataset.shape[1]):
            # denoised_channel = wavelet_denoise(channel)
            # denoised_dataset.append(denoised_channel)
            denoised_dataset[s, c] = wavelet_denoise(dataset[s, c])
    return denoised_dataset


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

scalers = get_scalers(x_test)

x_test_denoised = denoise(x_test)

x_test_normalized = normalization(x_test_denoised, scalers)

x_test_windows, y_test_windows = create_sliding_window(x_test_normalized, y_test, WINDOW_SIZE, STEP_SIZE, PRED_SIZE)

np.savez_compressed(
    path+'test_data_denoised.npz',
    x_test=x_test_windows,
    y_test=y_test_windows
)