#!/usr/bin/env python
# coding: utf-8

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




