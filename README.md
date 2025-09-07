# EEG Seizure Prediction with CNN-LSTM

<h3 align="center">

 <!-- Status -->
 <img alt="Status do Projeto" src="https://img.shields.io/badge/Status-Finished-lightgrey?style=for-the-badge&logo=headspace&logoColor=green&color=9644CD&labelColor=1C1E26">

 <!-- License -->
  <a href="./LICENSE" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/license%20-MIT-1C1E26?style=for-the-badge&labelColor=1C1E26&color=9644CD">
  </a>

</h3>

## Overview

This project implements a deep learning model for **seizure prediction from EEG signals**, using the [EEG Seizure Analysis Dataset](https://www.kaggle.com/datasets/adibadea/chbmitseizuredataset) based on [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/).

The system classifies whether a seizure will occur within the next 30 seconds, aiming to support clinical decision-making and early-warning systems for patients with epilepsy.

Our model combines **Convolutional Neural Networks (CNNs)** for feature extraction with **Long Short-Term Memory (LSTM)** networks for temporal modeling, providing robust performance on time-series EEG data.

---

## Features

- Preprocessing pipeline for multi-channel EEG signals
- 1DCNN–LSTM architecture with dropout and batch normalization
- Class imbalance handling with **pos_weight** in BCE loss
- Training with learning rate scheduling (ReduceLROnPlateau)
- Evaluation with accuracy, AUROC, F1-score, recall, and false negative rate
- Visualization of:
  - Training & validation accuracy/loss curves
  - Prediction probability distributions
  - Classification reports

---

## Dataset

- **Source:** [EEG Seizure Analysis Dataset](https://www.kaggle.com/datasets/adibadea/chbmitseizuredataset)
- **Description:** EEG recordings from pediatric subjects with intractable seizures.
- **Format:** Each sample is a window of EEG data with shape `(channels, timesteps)`.

---

## Model Architecture

- **CNN Layers:** Extract local spatial patterns from EEG channels
- **LSTM Layers:** Capture long-term temporal dependencies
- **Fully Connected Layers:** Classify preictal vs. non-preictal states

---

## Results

- **Validation Accuracy**: ~94.8%

- **Validation AUROC**: ~0.96

- **Validation F1-score**: ~0.83

- **False Negative Rate (FNR)**: ~0.13

---

## License

This template and the code in it is licensed under the [MIT License](https://github.com/marcizhu/readme-chess/LICENSE).
