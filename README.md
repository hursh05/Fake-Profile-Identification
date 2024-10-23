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


Usage
Data Exploration and Preprocessing:

Use the Jupyter notebook (notebooks/fake_profile_classification.ipynb) for interactive data exploration and cleaning.
Check for missing values and understand the distribution of different features using plots.
Training the Model:

Run the train_model.py script to train the ANN model on the training dataset:

python scripts/train_model.py

The trained model will be saved in the models/ directory as fake.h5.

Evaluating the Model:

Run the evaluate_model.py script to evaluate the trained model on the test dataset:

python scripts/evaluate_model.py

Model Architecture
The ANN model consists of the following layers:

Input Layer: 11 neurons (corresponding to the input features).
Hidden Layers:
First hidden layer: 50 neurons, ReLU activation.
Second hidden layer: 150 neurons, ReLU activation, with a 30% dropout.
Third hidden layer: 150 neurons, ReLU activation, with a 30% dropout.
Fourth hidden layer: 25 neurons, ReLU activation, with a 30% dropout.
Output Layer: 2 neurons (binary classification), softmax activation.
The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as the metric.

Results and Evaluation
Training History:

The training history shows the loss and accuracy progression over 50 epochs, including validation metrics.
Evaluation Metrics:

Accuracy: The model's overall classification accuracy on the test dataset.
Confusion Matrix: Visual representation of the model’s performance across the classes.
Classification Report: Detailed metrics including precision, recall, and F1-score.
Sample Results:

Example predictions on the test dataset, comparing the actual and predicted labels.

Contact
Author: Hursh Karnik
GitHub: hursh05
Email: hurhkarnik5603@gmail.com
