!pip install mne numpy pandas scikit-learn tensorflow

# Import required libraries
import os
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, LSTM, Bidirectional, GRU, Attention, BatchNormalization, Add, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import drive
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import RMSprop, Adamax
from scipy.signal import welch
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

drive.mount('/content/drive')

# Define paths to data directories
normal_dir = '/content/drive/MyDrive/NMT/nmt_scalp_eeg_dataset/normal/train'
abnormal_dir = '/content/drive/MyDrive/NMT/nmt_scalp_eeg_dataset/abnormal/train'
normal_eval_dir = '/content/drive/MyDrive/NMT/nmt_scalp_eeg_dataset/normal/eval'
abnormal_eval_dir = '/content/drive/MyDrive/NMT/nmt_scalp_eeg_dataset/abnormal/eval'

# Function to preprocess EEG data using mne
def preprocess_eeg_data(file_path, fixed_length=5000, sf=200):
    try:
        # Load the raw data
        raw = mne.io.read_raw_edf(file_path, preload=True)

        # Bandpass filter to retain frequencies between 1 and 40 Hz
        raw.filter(1, 40, fir_design='firwin')

        # Apply Independent Component Analysis (ICA) for artifact removal
        ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter=800)
        ica.fit(raw)
        ica.exclude = [0]  # Exclude first component (assuming it's an artifact like an eye blink)
        raw = ica.apply(raw)

        # Resample the data to 200 Hz
        raw.resample(sf)

        # Extract the data (e.g., first 10 channels)
        data = raw.get_data()[:10, :]

        # Ensure consistent length by truncating or padding
        if data.shape[1] > fixed_length:
            data = data[:, :fixed_length]  # Truncate to fixed_length
        elif data.shape[1] < fixed_length:
            # Pad with zeros to fixed_length
            padding = np.zeros((data.shape[0], fixed_length - data.shape[1]))
            data = np.concatenate((data, padding), axis=1)

        return data

    except Exception as e:
        print(f"An error occurred while processing file {file_path}: {e}")
        return None

# Function to compute PSD using Welch's method
def compute_psd(data, sf=200, nperseg=256):
    psd_features = []
    for channel in data:
        freqs, psd = welch(channel, sf, nperseg=nperseg)
        psd_features.append(psd)

    return np.array(psd_features).flatten()  # Flatten all channel PSDs into a single feature vector

# Function to load and preprocess EEG data from EDF files with manual label assignment
def load_and_preprocess_eeg_data(edf_directory, label, max_files=100, fixed_length=5000):
    X = []
    y = []
    file_count = 0  # Initialize counter for successfully processed files
    total_files = 0  # Counter to track total files attempted

    print(f"Loading data from directory: {edf_directory} with label {label}")

    for file_name in os.listdir(edf_directory):
        if file_name.startswith('._') or file_count >= max_files:
            continue

        if file_name.endswith('.edf'):
            total_files += 1  # Increment total files counter
            file_path = os.path.join(edf_directory, file_name)
            data = preprocess_eeg_data(file_path, fixed_length)

            if data is not None:
                # Compute PSD for each data sample
                psd_features = compute_psd(data)
                X.append(psd_features)  # Append PSD features
                y.append(label)
                file_count += 1  # Increment successfully processed files counter
                print(f"Processed file {file_count}/{max_files}: {file_name} with label {label}")
            else:
                print(f"An error occurred processing file {file_name}.")

    print(f"Total files attempted: {total_files}, Successfully processed: {file_count}")
    return np.array(X), np.array(y)

# Load training data with manual label assignment and preprocessing
X_normal_train, y_normal_train = load_and_preprocess_eeg_data(normal_dir, label=0, max_files=750)

X_abnormal_train, y_abnormal_train = load_and_preprocess_eeg_data(abnormal_dir, label=1, max_files=500)

# Convert to one-hot encoding format for training data
y_normal_train = to_categorical(y_normal_train, num_classes=2)
y_abnormal_train = to_categorical(y_abnormal_train, num_classes=2)

# Concatenate the training data and labels
X_train = np.concatenate((X_normal_train, X_abnormal_train), axis=0)
y_train = np.concatenate((y_normal_train, y_abnormal_train), axis=0)

# Load evaluation data with manual label assignment and preprocessing
X_normal_eval, y_normal_eval = load_and_preprocess_eeg_data(normal_eval_dir, label=0, max_files=500)

X_abnormal_eval, y_abnormal_eval = load_and_preprocess_eeg_data(abnormal_eval_dir, label=1, max_files=500)

# Convert to one-hot encoding format for evaluation data
y_normal_eval = to_categorical(y_normal_eval, num_classes=2)
y_abnormal_eval = to_categorical(y_abnormal_eval, num_classes=2)

# Concatenate the evaluation data and labels
X_eval = np.concatenate((X_normal_eval, X_abnormal_eval), axis=0)
y_eval = np.concatenate((y_normal_eval, y_abnormal_eval), axis=0)

# Analyze normal train data
def analyze_eeg_data(X_data, label, max_plots=5):
    print(f"Analyzing {label} data...")
    print(f"Shape: {X_data.shape}")
    print(f"Number of samples: {X_data.shape[0]}")

    # Visualize some of the data
    for i in range(min(max_plots, X_data.shape[0])):
        plt.figure(figsize=(15, 5))
        plt.title(f"{label} Sample {i + 1}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.plot(X_data[i])
        plt.show()

# Analyze normal and abnormal data
analyze_eeg_data(X_normal_train, "Normal")
analyze_eeg_data(X_abnormal_train, "Abnormal")

def plot_psd_comparison(data1, data2, sf=200, label1='Normal', label2='Abnormal'):
    """
    Plot the power spectral density (PSD) comparison between two datasets.

    :param data1: First dataset (normal).
    :param data2: Second dataset (abnormal).
    :param sf: Sampling frequency.
    :param label1: Label for the first dataset.
    :param label2: Label for the second dataset.
    """
    plt.figure(figsize=(15, 8))
    for i, channel_data in enumerate(data1[:5]):  # Visualize for the first 5 samples
        freqs, psd1 = welch(channel_data, sf, nperseg=256)
        freqs, psd2 = welch(data2[i], sf, nperseg=256)

        plt.subplot(5, 1, i + 1)
        plt.plot(freqs, psd1, label=label1)
        plt.plot(freqs, psd2, label=label2)
        plt.title(f"PSD Comparison for Sample {i+1} - Channel {i+1}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (dB/Hz)")
        plt.legend()

    plt.tight_layout()
    plt.show()

# Plot PSD comparison for normal and abnormal samples
plot_psd_comparison(X_normal_train, X_abnormal_train)

def plot_mean_std(data, label, color='blue'):
    """
    Plot mean and standard deviation across channels.

    :param data: EEG data.
    :param label: Label for the dataset.
    :param color: Color for the plots.
    """
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)

    plt.figure(figsize=(15, 5))
    plt.plot(mean_data, label=f'Mean - {label}', color=color)
    plt.fill_between(range(len(mean_data)), mean_data - std_data, mean_data + std_data, alpha=0.2, color=color)
    plt.title(f'Mean and Standard Deviation across Channels - {label}')
    plt.xlabel("Channel")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

# Plot mean and std for normal and abnormal data
plot_mean_std(X_normal_train, "Normal", color='green')
plot_mean_std(X_abnormal_train, "Abnormal", color='red')
