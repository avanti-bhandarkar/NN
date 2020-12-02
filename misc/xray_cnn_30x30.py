

import os
import tensorflow as tf
from google.colab.patches import cv2_imshow  
import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D , MaxPooling2D
from keras.models import Sequential 
import random
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from random import shuffle

os.chdir("/content/drive/My Drive/method1/")
images=np.load("images_small")
labels=np.load("labels_small")
ind=np.arange(len(images))
random.shuffle(ind)
images=[images[i] for i in ind]
labels=[labels[i] for i in ind]

x_tr= np.array(images[0:int(len(images)*0.6)])
label_tr= np.array(labels[0:int(len(images)*0.6)])
x_te= np.array(images[int(len(images)*0.6):])
label_te= np.array(labels[int(len(images)*0.6):])

num_labels=6

#print(x_tr.type)
label_trlin=label_tr
label_telin=label_te

x_tr = x_tr.reshape(1003,30,30,1)
x_te = x_te.reshape(669,30,30,1)

data = x_tr
labels = label_tr
x_te2 = x_te

#categorical data
label_tr = keras.utils.to_categorical(label_tr, num_labels)
label_te = keras.utils.to_categorical(label_te, num_labels)

#create model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(30,30,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#fitting model
history=model.fit(x_tr, label_tr, batch_size=128, epochs=10, validation_split=.2)

#evaluation
loss, accuracy=model.evaluate(x_tr,label_tr)
print(accuracy,loss)
print('\n')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.show()

print(f'cv loss: {loss:.3}')
print(f'cv accuracy: {accuracy:.3}')

#prediction

results=model.predict(x_te)
df = pd.DataFrame(results)
resultslin=(df.idxmax(axis=1))

#confusion matrix
con_mat = tf.math.confusion_matrix(labels=label_telin, predictions=resultslin).numpy()
print(con_mat)
print('\n')

#precision and recall 

con = pd.DataFrame(con_mat)
predpos=con.sum(axis = 0, skipna = True)
predneg=con.sum(axis = 1, skipna = True)
precision=np.zeros(num_labels)
recall=np.zeros(num_labels)
for i in range(num_labels):
  precision[i]=con_mat[i,i]/(predpos[i])
  recall[i]=con_mat[i,i]/(predneg[i])
print('precision',precision)
print('\n')
print('recall',recall)

# inference
index = int(input('Enter an index between 1 and 690 to test the network for'))
plt.imshow(x_te2[index].reshape(30,30),cmap='Greys')
#print('Our prediction is' , resultslin[index])

if resultslin[index] == 0:
  print('This does not seem to be a chest X-ray. Please upload the correct image.')
elif resultslin[index] == 1:
  print('This image matches our requirement. Do you wish to proceed')
elif resultslin[index] == 2:
  print('This chest X-ray has been rotated by 90 degrees. Please upload a non rotated image.')
elif resultslin[index] == 3:
  print('This chest X-ray has been rotated by 180 degrees. Please upload a non rotated image.')
elif resultslin[index] == 4:
  print('This chest X-ray has been rotated by 270 degrees. Please upload a non rotated image.')
elif resultslin[index] == 5:
  print('This does not seem to be an X-ray. Please upload the correct image.')
else:
  print('Unspecified error. Please recheck the image.')

from google.colab import drive
drive.mount('/content/drive')
