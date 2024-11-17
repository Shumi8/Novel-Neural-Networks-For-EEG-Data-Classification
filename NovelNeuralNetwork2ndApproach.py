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
X_normal_train, y_normal_train = load_and_preprocess_eeg_data(normal_dir, label=0, max_files=500)

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

# Scale the PSD features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_eval = scaler.transform(X_eval)

# Reshape the input data to add a channel dimension
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_eval_reshaped = X_eval.reshape(X_eval.shape[0], X_eval.shape[1], 1)

# Neural Network Model with Improved CNN and RNN layers for EEG Data
input_layer = Input(shape=(X_train_reshaped.shape[1], 1))  # Adding a channel dimension for Conv1D

# CNN Layers
conv1 = Conv1D(filters=64, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
conv1 = MaxPooling1D(pool_size=2)(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = Dropout(0.3)(conv1)  # Increased dropout for regularization

conv2 = Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001))(conv1)
conv2 = MaxPooling1D(pool_size=2)(conv2)
conv2 = BatchNormalization()(conv2)
conv2 = Dropout(0.4)(conv2)  # Increased dropout

conv3 = Conv1D(filters=256, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001))(conv2)
conv3 = MaxPooling1D(pool_size=2)(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = Dropout(0.5)(conv3)  # Further dropout increase

# RNN Layers
rnn_layer = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))(conv3)
rnn_layer = BatchNormalization()(rnn_layer)
rnn_layer = Dropout(0.4)(rnn_layer)  # Added dropout after LSTM

gru_layer = Bidirectional(GRU(32, return_sequences=False, kernel_regularizer=l2(0.001)))(rnn_layer)
gru_layer = BatchNormalization()(gru_layer)
gru_layer = Dropout(0.3)(gru_layer)  # Added dropout after GRU

# Fully Connected Dense Layers
dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(gru_layer)
dense1 = BatchNormalization()(dense1)
dense1 = Dropout(0.5)(dense1)

dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dense1)
dense2 = BatchNormalization()(dense2)
dense2 = Dropout(0.5)(dense2)

# Output Layer
output_layer = Dense(2, activation='softmax')(dense2)

# Compile the Model with Adam Optimizer
optimizer = Adam(learning_rate=0.0003)  # Adjusted learning rate

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model with Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=65, restore_best_weights=True)  # Increased patience
history = model.fit(X_train_reshaped, y_train, epochs=500, batch_size=32, validation_data=(X_eval_reshaped, y_eval), callbacks=[early_stopping])

# Evaluate the Model
loss, accuracy = model.evaluate(X_eval_reshaped, y_eval)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Make Predictions
predictions = model.predict(X_eval)
predicted_classes = np.argmax(predictions, axis=1)
true_classes_eval = np.argmax(y_eval, axis=1)

# Print out the Classification Report
print("\nClassification Report:")
print(classification_report(true_classes_eval, predicted_classes, target_names=['Class 0', 'Class 1']))

# Find indices of both classes (0 and 1)
indices_class_0 = np.where(true_classes_eval == 0)[0]
indices_class_1 = np.where(true_classes_eval == 1)[0]

# Print out 5 predictions for each class
print("Sample Predictions for Class 0:")
for i in indices_class_0[:5]:
    print(f"True Label: {true_classes_eval[i]}, Predicted Label: {predicted_classes[i]}, Prediction Confidence: {predictions[i]}")

print("\nSample Predictions for Class 1:")
for i in indices_class_1[:5]:
    print(f"True Label: {true_classes_eval[i]}, Predicted Label: {predicted_classes[i]}, Prediction Confidence: {predictions[i]}")