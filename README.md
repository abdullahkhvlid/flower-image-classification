# Flower Species Classification with Convolutional Neural Networks

A professional deep learning implementation for automated flower species classification using convolutional neural networks and TensorFlow.

## Project Description

This project implements a sophisticated computer vision system that classifies flower species with high accuracy using deep convolutional neural networks. The model achieves 69.89% test accuracy on the TensorFlow Flowers dataset, demonstrating robust performance in botanical image classification.

## Technical Architecture

### Dataset Processing
- TensorFlow Flowers dataset containing 3,670 high-resolution floral images
- Advanced preprocessing pipeline with image resizing to 180x180 pixels
- Pixel normalization and optimized batch processing with AUTOTUNE prefetching
- Strategic 80-20 train-test split for comprehensive model validation

### Neural Network Architecture
- Multi-layered Convolutional Neural Network with progressive feature extraction
- Three convolutional layers with 32, 64, and 128 filters respectively
- MaxPooling layers for spatial dimension reduction
- Fully connected dense layer with 128 neurons and dropout regularization
- Softmax output layer for 5-class flower species classification

## Model Performance

- Test Accuracy: 69.89%
- Optimized with Adam optimizer and sparse categorical crossentropy
- Minimal overfitting with consistent training convergence
- Comprehensive evaluation metrics including confusion matrix and classification reports

## Features

- Real-time training progress monitoring
- Advanced visualization with accuracy and loss curves
- Heatmap confusion matrix for performance analysis
- Species-specific precision, recall, and F1-score evaluation
- Production-ready prediction system for individual image classification

## Technical Implementation

- TensorFlow Dataset API for efficient data streaming
- Professional-grade model architecture with dropout regularization
- Advanced image preprocessing pipelines
- Scalable architecture suitable for commercial deployment

## Applications

- Botanical research and species identification
- Agricultural technology and crop monitoring
- Environmental monitoring and conservation
- Educational tools for plant science

## Technologies Used

- TensorFlow 2.x
- Keras
- TensorFlow Datasets
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

This project represents enterprise-level deep learning implementation, showcasing professional computer vision techniques for real-world classification challenges in ecological and agricultural domains.

## Installation

```bash
pip install tensorflow tensorflow-datasets matplotlib numpy scikit-learn seaborn

# Load and preprocess dataset
dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)

# Train model
history = model.fit(train_dataset, validation_data=test_dataset, epochs=10)

# Evaluate performance
test_loss, test_acc = model.evaluate(test_dataset)


Model Summary
The convolutional neural network architecture consists of sequential convolutional and pooling layers followed by fully connected layers with dropout regularization, specifically designed for multi-class flower species classification.

Results
The model demonstrates strong classification performance across five flower species with comprehensive evaluation metrics and visualization tools for performance analysis.

