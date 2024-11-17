# Novel Neural Network Architectures for EEG Data Classification

This repository contains the implementation of three innovative neural network architectures for EEG data classification. The project aims to improve the accuracy of detecting neurological disorders using the NMT Scalp EEG Dataset, focusing on capturing the spatial and temporal complexities of EEG signals.

## Features

### 1. First Neural Network Approach
- **Architecture**: Combination of fully connected layers, Convolutional Neural Networks (CNNs), and Gated Recurrent Units (GRUs).
- **Data Processing**:
  - Bandpass filtering of EEG signals (1-40 Hz).
  - Application of Independent Component Analysis (ICA) for artifact removal.
  - Resampling to 200 Hz and truncating or padding to fixed lengths.
- **Model Highlights**:
  - Extracts temporal features using GRUs.
  - Combines CNN layers to learn spatial features.
- **Performance**: Achieved moderate classification accuracy with balanced recall and precision.

### 2. Second Neural Network Approach
- **Architecture**: Advanced integration of CNN and Recurrent Neural Network (RNN) layers.
- **Enhancements**:
  - Improved feature extraction using multi-layer CNNs.
  - Temporal analysis using bidirectional LSTMs.
  - Employed dropout and batch normalization for regularization.
- **Model Evaluation**:
  - Improved performance over the first model, with detailed analysis of accuracy, F1-score, and ROC-AUC.

### 3. Third Neural Network Approach
- **Architecture**: Temporal Convolutional Network (TCN) combined with attention mechanisms.
- **Key Features**:
  - Captures long-range dependencies in EEG signals.
  - Attention layers focus on the most relevant features.
- **Performance Metrics**:
  - Best-performing model with the highest accuracy and reliability.
  - Detailed visualizations of accuracy-loss curves, confusion matrix, and ROC curves.

## Data Description
- **Dataset**: NMT Scalp EEG Dataset, containing labeled EEG recordings from healthy individuals and patients with neurological disorders.
- **Preprocessing**:
  - Normal vs. abnormal sample comparison using Power Spectral Density (PSD).
  - Feature extraction techniques such as Welch's method and Fourier Transform.
  - Oversampling techniques like SMOTE to address class imbalance.

## Technologies Used
- **Deep Learning Frameworks**: TensorFlow and Keras.
- **Signal Processing Libraries**: MNE, Scipy.
- **Evaluation Tools**:
  - Confusion matrices, ROC-AUC curves.
  - Classification metrics: accuracy, precision, recall, F1-score.
- **Visualization**: Matplotlib, Seaborn.

## Visualizations
- **PSD Analysis**:
  - Comparative visualizations of normal and abnormal EEG signals.
- **Model Performance**:
  - Accuracy and loss curves over epochs.
  - Confusion matrices to illustrate prediction quality.

## Usage
1. **Run Models**:
   - Execute individual scripts for each model architecture.
   - Provide the EEG dataset path in the configuration.

2. **Evaluate Results**:
   - Analyze classification reports and visualizations.

## Future Work
- Enhance spatial feature extraction using advanced CNN techniques.
- Introduce cross-dataset transfer learning to generalize models further.
- Explore additional preprocessing methods for artifact removal and feature extraction.
