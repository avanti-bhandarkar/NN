
import random
import cv2
from google.colab.patches import cv2_imshow
import glob
import os
import re
import numpy as np

def get_array(path,pat):
  os.chdir(path)
  name=[]
  curdir=os.listdir()
  for i in curdir:
    matches=pat.search(i)
    try:
      a = cv2.imread(matches.group())
      a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
      a=cv2.resize(a,(30,30),interpolation = cv2.INTER_AREA)
      a=cv2.normalize(a,a, 0, 255, cv2.NORM_MINMAX)
      name.append(a)
    except:
      continue
  #cv2_imshow(name[20])
  return name

pat=re.compile(".+\.(jpe?g$|png$|JPG)")

path = "/content/drive/My Drive/non chest x rays/"
namenon_chest=get_array(path,pat)
labnon_chest=[0]*len(namenon_chest)

path = "/content/drive/My Drive/chest no rot/"
namechest0=get_array(path,pat)
labchest0=[1]*len(namechest0)

path = "/content/drive/My Drive/chest 90/"
namechest90=get_array(path,pat)
labchest90=[2]*len(namechest90)

path = "/content/drive/My Drive/chest 180/"
namechest180=get_array(path,pat)
labchest180=[3]*len(namechest180)

path = "/content/drive/My Drive/chest 270/"
namechest270=get_array(path,pat)
labchest270=[4]*len(namechest270)

path="/content/drive/My Drive/random stuff/"
name_random=get_array(path,pat)
lab_random=[5]*len(name_random)


os.chdir("/content/drive/My Drive/")
name1=name_random+namenon_chest+namechest0+namechest90+namechest180+namechest270
label=lab_random+labnon_chest+labchest0+labchest90+labchest180+labchest270

file = open("images", "wb")
np.save(file, name1)
file.close
file = open("labels","wb")
np.save(file,label)
file.close

from google.colab import drive
drive.mount('/content/drive')
