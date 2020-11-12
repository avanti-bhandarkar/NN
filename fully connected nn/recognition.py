import tensorflow as tf
(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt

#image_index1 = 1000
#plt.imshow(x_tr[image_index1].reshape(28,28), cmap='Greys')
#print(y_tr[image_index1]) 

x_tr = x_tr.reshape(x_tr.shape[0], 28, 28, 1)
x_te = x_te.reshape(x_te.shape[0], 28, 28, 1)

ip_shape = (28, 28, 1)

x_tr = x_tr.astype('float32')
x_te = x_te.astype('float32')

x_tr /= 255
x_te /= 255

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='sgd', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy','categorical_accuracy'])

model.fit(x=x_tr,y=y_tr, epochs=6)

model.evaluate(x_te, y_te)

#inference script

image_index2 = 100
img_rows = img_cols = 28        
plt.imshow(x_te[image_index2].reshape(28,28),cmap='Greys')
pred = model.predict(x_te[image_index2].reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
