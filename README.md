# CNN for CIFAR-10: Hyperparameter Tuning

## Project Overview

This project investigates the impact of hyperparameter tuning on the performance of a Convolutional Neural Network (CNN) for classifying images in the CIFAR-10 dataset. While pretrained models like ResNet achieve high accuracy, this study uses a custom simple CNN as a baseline, optimizing its performance through systematic hyperparameter adjustments.

### Key Features:
- Focus on tuning epochs, batch size, kernel size, number of layers, and filters.
- Implementation of batch normalization for improved accuracy.
- Analysis of hyperparameter effects on accuracy and training time.

---

### Network Topology

![image](https://github.com/user-attachments/assets/7fd142cd-2470-405b-a1c4-418e3745a14f)

---

## Introduction

The CIFAR-10 dataset, consisting of 60,000 32x32 color images across 10 classes, provides a standard benchmark for image classification tasks. This project explores the role of hyperparameter optimization to achieve significant performance improvements with limited computational resources.

---

## Dataset

**CIFAR-10 Dataset:**
- 50,000 training images and 10,000 test images.
- Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

---

## Methodology

### Baseline Model
- **Architecture:** A simple CNN with:
  - Convolutional layers followed by ReLU activations.
  - Max-pooling layers for down-sampling.
  - Fully connected layers for classification.

### Hyperparameters Tuned
1. **Epochs and Batch Size:**
   - Initial tests identified optimal values to reduce overfitting while maintaining accuracy.

2. **Kernel Size:**
   - Tested sizes (3x3, 4x4, 5x5) based on literature.

3. **Number of Filters:**
   - Compared performance for 28, 32, and larger values.

4. **Number of Layers:**
   - Incrementally added layers to evaluate their effect on accuracy.

5. **Normalization:**
   - Introduced batch normalization for improved gradient propagation and regularization.


![image](https://github.com/user-attachments/assets/361b5d36-0c44-4760-9f6f-7c7ab3dd39bc)

---

## Hyperparameter Tuning Results

### Epochs and Batch Size
- **Batch Size:** Increasing from 64 to 128 improved efficiency but slightly reduced accuracy.
- **Epochs:** Optimal training at 29 epochs; training beyond this led to overfitting.

### Kernel Size
- **5x5 kernel:** Showed the best balance between training time and accuracy (based on Sinha et al., 2017).

### Number of Filters
- **32 filters:** Outperformed 28 filters, contrary to some prior findings, likely due to differences in model structure.

### Number of Layers
- Adding layers improved accuracy, with diminishing returns and increased risk of overfitting.
- Final model achieved optimal performance with an additional layer.

### Normalization
- Batch normalization yielded the most significant accuracy boost (from 0.7391 to 0.8160).

![image](https://github.com/user-attachments/assets/3c1c3cf6-6ec7-46ee-a8fb-d7488856c869)

![image](https://github.com/user-attachments/assets/10b6c934-a7a0-4e86-a26d-8e357ff3ab2c)

---

## Conclusion

The study demonstrates the importance of systematic hyperparameter tuning for CNNs:
- Adjusting kernel size, number of filters, and layers can lead to incremental improvements.
- Batch normalization proved critical for achieving higher accuracy.

Despite the limitations in computational resources, this study provides a framework for optimizing CNNs for CIFAR-10 classification. Future work could include automated hyperparameter search methods like grid search or particle swarm optimization.

---

## References

- Ahmed, W., & Karim, A. (2020). ‘The Impact of Filter Size and Number of Filters on Classification Accuracy in CNN.’
- Gong, J., et al. (2022). ‘ResNet10: A Lightweight Residual Network for Remote Sensing Image Classification.’
- Keskar, N., et al. (2023). ‘On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.’
- Tuba, E., et al. (2021). ‘Convolutional Neural Networks Hyperparameters Tuning.’
- Schilling, F. (2016). ‘The Effect of Batch Normalization on Deep Convolutional Neural Networks.’
