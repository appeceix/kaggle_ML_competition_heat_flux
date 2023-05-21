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
    
def preprocesamiento_train():
    pass

def preprocesamiento_test():
    pass

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