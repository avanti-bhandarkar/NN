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
import re

#dataset formation

def get_array(path,pat):
  os.chdir(path)
  name=[]
  curdir=os.listdir()
  for i in curdir:
    matches=pat.search(i)
    try:
      a = cv2.imread(matches.group())
      a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
      ret,a = cv2.threshold(a,0,255,cv2.THRESH_BINARY_INV)
      a=cv2.resize(a,(50,50),interpolation = cv2.INTER_AREA)
      a=cv2.normalize(a,a, 0, 255, cv2.NORM_MINMAX)
      name.append(a)
    except:
      continue
  #cv2_imshow(name[20])
  return name

pat=re.compile(".+\.(png)")

path = "/content/drive/My Drive/hindi classification/num/0"
dig0=get_array(path,pat)
lab0=[0]*len(dig0)

path = "/content/drive/My Drive/hindi classification/num/1"
dig1=get_array(path,pat)
lab1=[1]*len(dig1)

path = "/content/drive/My Drive/hindi classification/num/2"
dig2=get_array(path,pat)
lab2=[2]*len(dig2)

path = "/content/drive/My Drive/hindi classification/num/3"
dig3=get_array(path,pat)
lab3=[3]*len(dig3)

path = "/content/drive/My Drive/hindi classification/num/4"
dig4=get_array(path,pat)
lab4=[4]*len(dig4)

path = "/content/drive/My Drive/hindi classification/num/5"
dig5=get_array(path,pat)
lab5=[5]*len(dig5)

path = "/content/drive/My Drive/hindi classification/num/6"
dig6=get_array(path,pat)
lab6=[6]*len(dig6)

path = "/content/drive/My Drive/hindi classification/num/7"
dig7=get_array(path,pat)
lab7=[7]*len(dig7)

path = "/content/drive/My Drive/hindi classification/num/8"
dig8=get_array(path,pat)
lab8=[8]*len(dig8)

path = "/content/drive/My Drive/hindi classification/num/9"
dig9=get_array(path,pat)
lab9=[9]*len(dig9)

os.chdir("/content/drive/My Drive/")
name_hindi= dig0+dig1+dig2+dig3+dig4+dig5+dig6+dig7+dig8+dig9
label_hindi=lab0+lab1+lab2+lab3+lab4+lab5+lab6+lab7+lab8+lab9

file = open("images_hindi", "wb")
np.save(file, name_hindi)
file.close
file = open("labels_hindi","wb")
np.save(file,label_hindi)
file.close

#convolutional neural network

os.chdir("/content/drive/My Drive/")
images=np.load("images_hindi")
labels=np.load("labels_hindi")
ind=np.arange(len(images))
random.shuffle(ind)
images=[images[i] for i in ind]
labels=[labels[i] for i in ind]

x_tr= np.array(images[0:int(len(images)*0.6)])
label_tr= np.array(labels[0:int(len(images)*0.6)])
x_te= np.array(images[int(len(images)*0.6):])
label_te= np.array(labels[int(len(images)*0.6):])

num_labels=10

#print(x_tr.type)
label_trlin=label_tr
label_telin=label_te

x_tr = x_tr.reshape(x_tr.shape[0],50,50,1)
x_te = x_te.reshape(x_te.shape[0],50,50,1)

data = x_tr
labels = label_tr
x_te2 = x_te

#categorical data
label_tr = keras.utils.to_categorical(label_tr, num_labels)
label_te = keras.utils.to_categorical(label_te, num_labels)

#create model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(50,50,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#fitting model
history=model.fit(x_tr, label_tr, batch_size=128, epochs=3, validation_split=.2)

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
index = int(input('Enter an index between 1 and  to test the network for'))
plt.imshow(x_te2[index].reshape(50,50),cmap='Greys')

if resultslin[index] == 0:
  print('This is the Devanagari number 0')
elif resultslin[index] == 1:
  print('This is the Devanagari number 1')
elif resultslin[index] == 2:
  print('This is the Devanagari number 2')
elif resultslin[index] == 3:
  print('This is the Devanagari number 3')
elif resultslin[index] == 4:
  print('This is the Devanagari number 4')
elif resultslin[index] == 5:
  print('This is the Devanagari number 5')
elif resultslin[index] == 6:
  print('This is the Devanagari number 6')
elif resultslin[index] == 7:
  print('This is the Devanagari number 7')
elif resultslin[index] == 8:
  print('This is the Devanagari number 8')
elif resultslin[index] == 9:
  print('This is the Devanagari number 9')
else:
  print('This does not appear to be a Devanagari number. Please recheck the input')
