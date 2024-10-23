import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,accuracy_score,roc_curve,confusion_matrix
import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import StandardScaler
import Fake_Profile_Identification


model = tf.keras.models.load_model('D:\project\Fake profile\ANN Mini Project 2\ifake.h5')
# Create a StandardScaler for scaling input data
scaler = StandardScaler()

def predict_fake_profile():
    inputs = [float(entry.get()) for entry in entry_widgets]

    # Ensure the scaler is fitted with training data
    # Assuming X_train is the training data used to train the model
    # If X_train is not available, replace it with the appropriate training data
    X_train = Fake_Profile_Identification.X_train 
    scaler.fit(X_train)

    # Scale the input data using the fitted scaler
    scaled_inputs = scaler.transform([inputs])

    # Make predictions using the trained model
    prediction = model.predict(np.array(scaled_inputs))  # Convert to numpy array

    # Display the prediction result
    result_label.config(text=f"Prediction: {'Fake' if prediction[0][1] > 0.5 else 'Real'}")

# Create the main window
root = tk.Tk()
root.title("Fake Profile Detection")

# Create labels and entry widgets for user input
attributes = ['nums/length username', 'fullname words', 'nums/length fullname', 'name==username', 'description length',
              'external URL', 'private', '#posts', '#followers', '#follows']

label_widgets = []
entry_widgets = []

for i, attribute in enumerate(attributes):
    label = ttk.Label(root, text=attribute)
    entry = ttk.Entry(root)
    
    label.grid(row=i, column=0, padx=10, pady=5, sticky='w')
    entry.grid(row=i, column=1, padx=10, pady=5)
    
    label_widgets.append(label)
    entry_widgets.append(entry)

# Create a button for prediction
predict_button = ttk.Button(root, text="Predict", command=predict_fake_profile)
predict_button.grid(row=len(attributes), column=0, columnspan=2, pady=10)

# Create a label for displaying the prediction result
result_label = ttk.Label(root, text="Prediction: ")
result_label.grid(row=len(attributes)+1, column=0, columnspan=2, pady=5)

# Run the Tkinter event loop
root.mainloop()