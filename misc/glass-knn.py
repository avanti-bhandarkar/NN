import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

df=pd.read_csv('/content/glass.csv')

X = df.drop('Type',axis=1).values
y = df['Type'].values
h = 0.2

#K-nearest neighbour model

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

neighbors = np.arange(1,20)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy[i] = knn.score(X_train, y_train)

    test_accuracy[i] = knn.score(X_test, y_test) 

plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)

from sklearn.metrics import confusion_matrix

y_predict = knn.predict(X_test)
confusion_matrix(y_test,y_predict)
#or use the code below
pd.crosstab(y_test, y_predict, rownames=['True'], colnames=['Predicted'], margins=True)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,zero_division=1))

#boxplot for the refractive index and class of glass

sns.boxplot('Type', 'RI', data = df)

