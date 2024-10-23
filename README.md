# Fake-Profile-Identification



This project uses an Artificial Neural Network (ANN) to classify Instagram profiles as either "fake" or "real." The dataset contains various features about the profiles, such as whether they have a profile picture, whether their account is private, and the length of their username. The ANN model is trained to learn patterns and make predictions based on these features.

# Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Results and Evaluation](#results-and-evaluation)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

## Overview

The goal of this project is to use machine learning techniques, specifically an ANN, to identify fake profiles on Instagram. The project involves:
- Data loading and preprocessing.
- Exploratory data analysis (EDA) for understanding the dataset.
- Building and training an ANN model using TensorFlow and Keras.
- Evaluating the model’s performance with metrics such as accuracy, confusion matrix, and classification report.

## Project Structure

The repository is organized as follows:
- `data/`: Contains the training and testing datasets.
  - `train.csv`: Training data with features and labels.
  - `test.csv`: Testing data for evaluating model performance.
- `models/`: Directory to save the trained models.
  - `fake.h5`: Saved ANN model used for classification.
- `scripts/`: Python scripts used for data processing, model training, and evaluation.
  - `train_model.py`: Script for training the ANN model.
  - `evaluate_model.py`: Script for evaluating the trained model on the test dataset.
- `notebooks/`: Jupyter notebooks for interactive exploration and testing.
  - `fake_profile_classification.ipynb`: Contains all the code for data analysis, model training, and evaluation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-profile-identification-ann.git
   cd fake-profile-identification-ann
