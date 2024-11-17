!pip install mne h5py tensorflow imbalanced-learn

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, GRU, Flatten, LeakyReLU, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from google.colab import drive
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from tensorflow.keras.regularizers import l2

drive.mount('/content/drive')

# Load data
normal_train = np.load('/content/drive/MyDrive/NMT/normal_train_data2.npy')
abnormal_train = np.load('/content/drive/MyDrive/NMT/abnoraml_train_data.npy')
normal_eval = np.load('/content/drive/MyDrive/NMT/normal_eval_data.npy')
abnormal_eval = np.load('/content/drive/MyDrive/NMT/abnormal_eval_data.npy')

# Combine normal and abnormal data for training and evaluation
normal_train_dim = normal_train.shape[-1]
normal_train_zeros = np.zeros(normal_train_dim)

abnormal_train_dim = abnormal_train.shape[-1]
abnormal_train_ones = np.ones(abnormal_train_dim)

train_data = np.dstack((normal_train, abnormal_train))
train_label = np.append(normal_train_zeros, abnormal_train_ones)
train_data = np.swapaxes(train_data, 0, 2)

bs, t, f = train_data.shape
train_label = tf.keras.utils.to_categorical(train_label, num_classes=2)

normal_eval_dim = normal_eval.shape[-1]
normal_eval_zeros = np.zeros(normal_eval_dim)

abnormal_eval_dim = abnormal_eval.shape[-1]
abnormal_eval_ones = np.ones(abnormal_eval_dim)

eval_data = np.dstack((normal_eval, abnormal_eval))
eval_label = np.append(normal_eval_zeros, abnormal_eval_ones)
eval_data = np.swapaxes(eval_data, 0, 2)

eval_label = tf.keras.utils.to_categorical(eval_label, num_classes=2)

print('Data and labels have been loaded and processed.')
print('Training data shape:', train_data.shape)
print('Training labels shape:', train_label.shape)
print('Evaluation data shape:', eval_data.shape)
print('Evaluation labels shape:', eval_label.shape)

def extract_features(data):
    features = []
    for sample in data:
        mean_features = np.mean(sample, axis=1)
        std_features = np.std(sample, axis=1)
        skew_features = skew(sample, axis=1)
        kurt_features = kurtosis(sample, axis=1)
        fft_features = np.abs(fft(sample, axis=1))
        sample_features = np.concatenate((mean_features, std_features, skew_features, kurt_features, fft_features.flatten()))
        features.append(sample_features)
    return np.array(features)

# Extract features from training and evaluation data
train_features = extract_features(train_data)
eval_features = extract_features(eval_data)

print('Extracted features shape (train):', train_features.shape)
print('Extracted features shape (eval):', eval_features.shape)

# Standardize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
eval_features = scaler.transform(eval_features)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(train_label, axis=1)), y=np.argmax(train_label, axis=1))
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights:", class_weights_dict)

# Handle imbalance using SMOTE
smote = SMOTE(sampling_strategy='minority', random_state=42)
train_features_resampled, train_label_resampled = smote.fit_resample(train_features, np.argmax(train_label, axis=1))

train_label_resampled = tf.keras.utils.to_categorical(train_label_resampled, num_classes=2)

# Define model with increased complexity and adjusted regularization
input_layer = Input(shape=(train_features_resampled.shape[1],))
x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Reshape((32, 2))(x)
x = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.4)(x)

x = GRU(32, return_sequences=False, kernel_regularizer=l2(1e-4))(x)
x = Dense(16, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

output_layer = Dense(2, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks with less aggressive learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-5)

# Train the model
hist = model.fit(
    train_features_resampled,
    train_label_resampled,
    epochs=200,
    batch_size=64,
    verbose=1,
    shuffle=True,
    class_weight=class_weights_dict,
    validation_split=0.2,
    callbacks=[reduce_lr]
)

# Evaluate the model
test_score = model.evaluate(eval_features, eval_label, batch_size=32)
print("Evaluation loss and accuracy after training:", test_score)

# Predict and generate classification report
predictions = model.predict(eval_features)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(eval_label, axis=1)
report = classification_report(true_classes, predicted_classes, target_names=['Class 0', 'Class 1'])
print(report)
