import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report

from sklearn.exceptions import ConvergenceWarning
from matplotlib.colors import ListedColormap
from decimal import Decimal
from sklearn import svm
from sklearn import metrics

from sklearn.metrics import recall_score

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

estudiantes = pd.read_csv('estudiantes.csv')

# remplaza donde no halla valores con 0
bd=estudiantes.replace(np.nan,"0")

#estos valores de las columnas no son numeros, sino palabras, asi que se convierten en numeros

bd['genero']=encoder.fit_transform(bd.gender.values)

aux=bd.sort_values('genero')

x=aux.loc[:,['math score','reading score','writing score']]
#va a clasificar por genero
y=aux.loc[:,['genero']]


#se divide la muestra de entrenamiento y de test
x_train,x_test,y_train,y_test = train_test_split(x,y)

#y_train=y_train.values.ravel()

#se estandarizan los datos
scaler= StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)
print("-----------------")
print(np.shape(x_train),np.shape(x_test))
print("-----------------")

plt.scatter(aux.iloc[0:499,0], aux.iloc[0:499,1], color='red', marker ='o', label='setosa')
plt.scatter(aux.iloc[500:1000,0], aux.iloc[500:1000,1], color='blue', marker ='x', label='versicolor')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

mlp=MLPClassifier( hidden_layer_sizes=(3,9),
    max_iter=100,
    solver="sgd",
    verbose=10,
    random_state=33,
    learning_rate_init=0.001,
    activation= "logistic",
    )


#se entrena la red
mlp.fit(x_train,y_train)

#se testea la red
predictions=mlp.predict(x_test)
print("-----------------------")
print(predictions)
print("-----------------------")

y_predict = mlp.predict(x_test)

print(classification_report(y_test,predictions))


print("Training set score: %f" % mlp.score(x_train, y_train))
print("Test set score: %f" % mlp.score(x_test, y_test))

'''
plot_decision_regions(x_test, predictions, classifier=mlp)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.show()
'''