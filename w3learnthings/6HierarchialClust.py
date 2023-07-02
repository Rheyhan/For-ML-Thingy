"""
Hierarchical Clustering:

unsupervised learning method for clustering data points. The algorithm builds clusters
by measuring the dissimilarities between data. Unsupervised learning means that a model
does not have to be trained, and we do not need a "target" variable. 

Hierarchical clustering requires us to decide on both a distance and linkage method.
"""

#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

df=pd.DataFrame({
    "x":[4, 5, 10, 4, 3, 11, 14 , 6, 10, 12],
    "y":[21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
})
data=list(zip(df.x,df.y))

def fordendro(data):
    linkage_data = linkage(data, method='ward', metric='euclidean')
    dendrogram(linkage_data)
    plt.show()    #n=2
     
def forhierarchical(data):
    hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    labels = hierarchical_cluster.fit_predict(data)
    plt.scatter(df.x,df.y,c=labels)         
    plt.show()
    
if __name__ =="__main__":
    plt.scatter(df.x,df.y)                                  #Normal Scatter Plot
    plt.show()
    
    fordendro(data)                                         #Visualize Dendogram
    
    forhierarchical(data)                            #Visualize after clustering            