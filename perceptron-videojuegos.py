
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import Perceptron

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report


import warnings
warnings.filterwarnings('ignore')

encoder=LabelEncoder()

juegos = pd.read_csv('videojuegos.csv')


# remplaza donde no halla valores con 0
juegos=juegos.replace(np.nan,"0")


juegos["Platform"]=juegos["Platform"].replace("2600","Atari")
encoder=LabelEncoder()

#estos valores de las columnas no son numeros, sino palabras, asi que se convierten en numeros
juegos['plataforma']=encoder.fit_transform(juegos.Platform.values)
juegos['publica']=encoder.fit_transform(juegos.Publisher.values)
juegos['genero']=encoder.fit_transform(juegos.Genre.values)

aux=juegos.sort_values('genero')
#aux.to_csv('juegos2.csv',sep='\t')
#obtenemos las caracteristicas que necesitamos
aux2=aux.loc[0:3367,['plataforma','publica']]
#aux2.to_csv('juegos3.csv',sep='\t')

x=aux.loc[3267:3367,['plataforma','publica']]
#va a clasificar por genero
#y=aux.loc[3267:3367,['genero']]
y=aux.loc[:170,['genero']]
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y)

#y_train=y_train.values.ravel()


#se estandarizan los datos
scaler= StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)

'''
plt.plot(x_train, y_train, color='red', linewidth=1)
plt.title('Regresión Lineal Simple')
plt.xlabel('x')
plt.ylabel('y')
plt.show()  3267:3316,]
'''
plt.scatter(aux.loc[3267:3316,['plataforma','publica']], aux.loc[3267:3316,['plataforma','publica']], color='red', marker ='o', label='setosa')
plt.scatter(aux.loc[3317:3367,['plataforma','publica']], aux.loc[3317:3367,['plataforma','publica']], color='blue', marker ='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc="upper left")
plt.show()

per = Perceptron( eta0=0.01,random_state=12, max_iter=100, verbose=1, shuffle=True)
per.fit(x_train,y_train)
predictions = per.predict(x_test)
print("-----------------------")
print(predictions)

print(classification_report(y_test,predictions))


print("Training set score: %f" % per.score(x_train, y_train))
print("Test set score: %f" % per.score(x_test, y_test))

