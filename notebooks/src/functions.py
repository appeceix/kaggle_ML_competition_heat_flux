import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


def heatmap(dataset, label = None):
    import matplotlib.pyplot as plt
    corr = dataset.corr(method = 'spearman')
    plt.figure(figsize = (10, 10), dpi = 300)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask = mask, cmap = 'viridis', annot = True, annot_kws = {'size' : 12})
    plt.title(f'Matriz de correlación de {label} \n', fontsize = 25, weight = 'bold')
    plt.show()

def distancia(data, label = ''):
    from scipy.spatial.distance import squareform
    corr = data.corr(method = 'spearman')
    dist_linkage = linkage(squareform(1 - abs(corr)), 'complete')
    
    plt.figure(figsize = (10, 8), dpi = 300)
    dendro = dendrogram(dist_linkage, labels=data.columns, leaf_rotation=90)
    plt.title(f'Distancia entre variables {label}', weight = 'bold', size = 22)
    plt.show()

class ImputadorCategoricas(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, x, y = None):
        return self
    def transform(self, x, y = None):
        x_copy = x.copy()
        x_copy['author'] = x_copy['author'].fillna('Thompson')
        x_copy['geometry'] = x_copy['geometry'].fillna('tube')
        return x_copy
    
def plot_learning_curve(estimator, X, y, cv, train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='neg_mean_squared_error'
    )
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Entrenamiento')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Prueba')
    plt.fill_between(
        train_sizes, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1, color='r'
    )
    plt.fill_between(
        train_sizes, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1, color='g'
    )
    plt.xlabel('Tamaño del conjunto de entrenamiento')
    plt.ylabel('Pérdida')
    plt.title('Curva de aprendizaje')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def csv_datos(nombre_archivo, data):
    '''Guarda el dataframe data en un archivo csv con nombre nombre_archivo
    en la carpeta data/processed'''
    ruta_archivo = '../data/processed'
    data.to_csv(ruta_archivo + "/" + nombre_archivo, sep='\t', index=False)

def guardar_modelo(modelo, nombre_archivo):
    import pickle
    try:
        with open(nombre_archivo, 'wb') as archivo:
            pickle.dump(modelo, archivo)
        print(f"El modelo se ha guardado exitosamente en {nombre_archivo}.")
    except IOError:
        print("Error: No se pudo guardar el modelo. Permiso denegado.")