
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import Perceptron

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
import warnings

def plot_decision_regions(X, y, classifier, resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #Plot the decision surface
    
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl,1], alpha = 0.8, c = colors[idx], marker = markers[idx], label = cl, edgecolor  = 'black')
 
warnings.filterwarnings('ignore')

encoder=LabelEncoder()

iris = pd.read_csv('iris.data')

iris['target']=encoder.fit_transform(iris.label.values)

x=iris.loc[:99,['f1','f2',]]
#va a clasificar por genero
y=iris.loc[:99,['target']]

print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y)


#se estandarizan los datos
scaler= StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)

per = Perceptron(eta0=.0005,tol=1e-3, random_state=0, max_iter=3000, verbose=1, shuffle=False)
per.fit(x, y)
predictions = per.predict(x_test)
print("-----------------------")
print(predictions)

print(classification_report(y_test,predictions))

print("Training set score: %f" % per.score(x_train, y_train))
print("Test set score: %f" % per.score(x_test, y_test))

plot_decision_regions(x_test, predictions, classifier=per)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.show()
