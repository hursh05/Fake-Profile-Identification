
# In[1]:


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

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


instagram_df_train=pd.read_csv('D:/project/Fake profile/ANN Mini Project 2/archive/train.csv')
instagram_df_train


# In[5]:


instagram_df_test=pd.read_csv('D:/project/Fake profile/ANN Mini Project 2/archive/test.csv')
instagram_df_test


# In[6]:


instagram_df_train.head()


# In[7]:


instagram_df_train.tail()


# In[8]:


instagram_df_train.info()


# In[9]:


instagram_df_train.describe()


# In[10]:


instagram_df_train.isnull().sum()


# In[11]:


instagram_df_train['profile pic'].value_counts()


# In[12]:


instagram_df_train['fake'].value_counts()


# In[13]:


sns.countplot(x='fake', data=instagram_df_train)
plt.show()


# In[14]:


sns.countplot(x='private', data=instagram_df_train)
plt.show()


# In[15]:


sns.countplot(x='profile pic', data=instagram_df_train)
plt.show()


# In[16]:


plt.figure(figsize=(20, 10))
sns.displot(instagram_df_train['nums/length username'])
plt.show()


# In[17]:


plt.figure(figsize=(20, 20))
cm = instagram_df_train.corr()
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)
plt.show()


# In[18]:


X_train = instagram_df_train.drop(columns = ['fake'])
X_test = instagram_df_test.drop(columns = ['fake'])
X_train


# In[19]:


y_train = instagram_df_train['fake']
y_test = instagram_df_test['fake']
y_train


# In[20]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)


# In[21]:


y_train = tf.keras.utils.to_categorical(y_train, num_classes = 2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 2)


# In[22]:


y_train


# In[23]:


import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential(name = "fake")
model.add(Dense(50, input_dim=11, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation='softmax'))

model.summary()


# In[24]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[25]:


model.save('D:/project/Fake profile/ANN Mini Project 2/fake.h5')


# In[26]:


epochs_hist = model.fit(X_train, y_train, epochs = 50,  verbose = 1, validation_split = 0.1)


# In[27]:


# In[ ]:


print(epochs_hist.history.keys())


# In[ ]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()


# In[ ]:


predicted = model.predict(X_test)


# In[ ]:


predicted_value = []
test = []
for i in predicted:
    predicted_value.append(np.argmax(i))
    
for i in y_test:
    test.append(np.argmax(i))


# In[ ]:


print(classification_report(test, predicted_value))


# In[ ]:


plt.figure(figsize=(10, 10))
cm=confusion_matrix(test, predicted_value)
sns.heatmap(cm, annot=True)
plt.show()


# In[ ]:


import tkinter as tk
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Define paths
model_path = "D:/project/Fake profile/ANN Mini Project 2/fake.h5"
real_image_path = "D:/project/Fake profile/ANN Mini Project 2/real.png"
fake_image_path = "D:/project/Fake profile/ANN Mini Project 2/fake.png"

# Load the model
model = load_model(model_path)

# Define feature labels and entries
feature_labels = [
   "profile pic",
  "Username Length:",
  "Fullname Words:",
  "Fullname Length:",
  "Username in Fullname (0/1):",
  "Description Length:",
  "External URL (0/1):",
  "Private (0/1):",
  "# Posts:",
  "# Followers:",
  "# Follows:",
]

feature_entries = [tk.Entry(width=10) for _ in feature_labels]

# Create the Tkinter window
window = tk.Tk()
window.title("Fake Instagram Account Detector")

# Add labels and entries for each feature
for i, label in enumerate(feature_labels):
  tk.Label(text=label).grid(row=i, column=0)
  feature_entries[i].grid(row=i, column=1)

# Initialize prediction label and image
prediction_label = tk.Label(text="Prediction:")
prediction_image = tk.PhotoImage(file=real_image_path)
prediction_image_label = tk.Label(image=prediction_image)

# Add prediction label and image to grid
prediction_label.grid(row=len(feature_labels), column=0)
prediction_image_label.grid(row=len(feature_labels), column=1)

# Define scaler object and fit it on training data
scaler_x = StandardScaler()
scaler_x.fit(X_train)

def predict_click():
  try:
    # Get user inputs
    inputs = [float(entry.get()) for entry in feature_entries]

    # Validate input lengths
    if len(inputs) != len(feature_labels):
      raise ValueError("Incorrect number of inputs")

    # Scale the input data
    x_input = scaler_x.transform([inputs])

    # Predict
    prediction = model.predict(x_input)

    # Get the predicted class
    predicted_class = np.argmax(prediction)

    # Update the label and image based on prediction
    if predicted_class == 0:
      prediction_label['text'] = "Account is Real "
      prediction_image = tk.PhotoImage(file=real_image_path)
    else:
      prediction_label['text'] = "Account is Fake "
      prediction_image = tk.PhotoImage(file=fake_image_path)

    prediction_image_label.configure(image=prediction_image)

  except ValueError as e:
    prediction_label['text'] = f"Error: {e}"


# Add button to trigger prediction
predict_button = tk.Button(text="Predict", command=predict_click)
predict_button.grid(row=len(feature_labels) + 1, column=0, columnspan=2)

# Run the window main loop
window.mainloop()
