import tensorflow as tf
  
import keras
from keras.datasets import mnist 
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.models import Sequential 

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

n=784
num_labels=10

#prepocessing data

(x_tr, label_tr), (x_te, label_te) = mnist.load_data()
#print(x_tr.type)
label_trlin=label_tr
label_telin=label_te

x_tr = x_tr.reshape(60000,28,28,1)
x_te = x_te.reshape(10000,28,28,1)

data = x_tr
labels = label_tr
x_te2 = x_te

#categorical data
label_tr = keras.utils.to_categorical(label_tr, num_labels)
label_te = keras.utils.to_categorical(label_te, num_labels)

#create model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#fitting model
es_callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
history=model.fit(x_tr, label_tr, batch_size=128, epochs=8, validation_split=.5,callbacks = es_callback)

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

print('\n')
index = int(input('Enter an index between 1 and 10000 to test the network for'))
plt.imshow(x_te2[index].reshape(28,28),cmap='Greys')
print('prediction is ' , resultslin[index])
