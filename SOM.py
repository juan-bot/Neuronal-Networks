from minisom import MiniSom
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
def classify(som, data):

    winmap = som.labels_map(X_train, y_train)
    print(winmap)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

warnings.filterwarnings('ignore')

encoder=LabelEncoder()

estudiantes = pd.read_csv('estudiantes.csv')

# remplaza donde no halla valores con 0
bd=estudiantes.replace(np.nan,"0")

#estos valores de las columnas no son numeros, sino palabras, asi que se convierten en numeros

bd['genero']=encoder.fit_transform(bd.gender.values)

aux=bd.sort_values('genero')

x=aux.loc[:,['math score','reading score']]
#va a clasificar por genero
y=aux.loc[:,['genero']]
x=np.array(x)

X_train, X_test, y_train, y_test = train_test_split(x, y)


som = MiniSom(7, 7, 2, sigma=2, learning_rate=0.5, neighborhood_function='triangle', random_seed=10)

#som.train(x, 500, verbose=True)


#som.pca_weights_init(X_train)
som.train_batch(X_train, 100, verbose=True)
#print(y_test)
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(i) for i in X_train]).T
print(winner_coordinates)
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, (7,7))


# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(X_train[cluster_index == c, 0],
                X_train[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting centroids
for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                s=2, linewidths=5, color='k', label='centroid')
plt.legend();
#print(classification_report(y_test, classify(som, X_test)))

#print("Training set score: %f" % som.quantization(X_test))
#print("Test set score: %f" % som.score(X_test, y_test))
