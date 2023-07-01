'''
Data Generation
From ML_BerndKlein
'''

"""GENERATORS FOR CLASSIFICATION AND CLUSTERING"""

#libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import sklearn.datasets as ds

class blob:                                 #Generate isotropic Gaussian blobs for clustering.
    def randomcenter():
        #random generator
        n_classes = 4
        data, labels = make_blobs(n_samples=1000, centers=n_classes, random_state=100)

        #visualizing
        fig, ax = plt.subplots()
        colours=("green", "orange", "blue", "pink")
        for label in range(n_classes):
            ax.scatter(x=data[labels==label, 0],
                    y=data[labels==label, 1], 
                    c=colours[label], s=40, 
                    label=label)
        ax.set(xlabel='X', ylabel='Y', title='Blobs Examples')
        ax.legend(loc='upper right')
        plt.show()
        
    def chosencenter():
        #cluster location
        centers = [[2, 3], [4, 5], [7, 9]]
        #random generator
        data, labels = make_blobs(n_samples=1000,
                                centers=np.array(centers),
                                random_state=1)
        #visualizing
        fig, ax = plt.subplots()
        colours=("green", "orange", "blue")
        for label in range(len(centers)):
            ax.scatter(x=data[labels==label, 0],
                    y=data[labels==label, 1], 
                    c=colours[label], s=40, 
                    label=label)
        ax.set(xlabel='X', ylabel='Y', title='Blobs Examples')
        ax.legend(loc='upper right')
        plt.show()

class makemoon:
    def normalmoon():                             #Make two interleaving half circles.
        #random generator
        data, labels = ds.make_moons(n_samples=150, 
                                    shuffle=True, noise=0.19, 
                                    random_state=None)
        data += np.array(-np.ndarray.min(data[:,0]),
                        -np.ndarray.min(data[:,1]))
        
        #visualizing        
        fig, ax = plt.subplots()
        ax.scatter(data[labels==0, 0], data[labels==0, 1], 
                    c='orange', s=40, label='oranges')
        ax.scatter(data[labels==1, 0], data[labels==1, 1], 
                    c='blue', s=40, label='blues')
        ax.set(xlabel='X', ylabel='Y', title='Moons')
        ax.legend(loc="upper right")
        plt.show()
    
    def scaledmoon():
        #function for the scale
        def scale_data(data, new_limits, inplace=False ):
            if not inplace:
                data = data.copy()
                min_x, min_y = np.ndarray.min(data[:,0]), np.ndarray.min(data[:,1])
                max_x, max_y = np.ndarray.max(data[:,0]), np.ndarray.max(data[:,1])
                min_x_new, max_x_new = new_limits[0]
                min_y_new, max_y_new = new_limits[1]
                data -= np.array([min_x, min_y])
                data *= np.array([(max_x_new - min_x_new) / (max_x - min_x),
                                  (max_y_new - min_y_new) / (max_y - min_y)])
                data += np.array([min_x_new, min_y_new])
            if inplace:
                return None
            else:
                return data
        #random Gen
        data, labels = ds.make_moons(n_samples=100, shuffle=True, 
                                     noise=0.05, random_state=None)
        scale_data(data, [(1,4), (3,8)], inplace=True)
        
        #Visualizing
        fig, ax = plt.subplots()
        ax.scatter(data[labels==0, 0], data[labels==0, 1], 
                    c='orange', s=40, label='oranges')
        ax.scatter(data[labels==1, 0], data[labels==1, 1], 
                    c='blue', s=40, label='blues')
        ax.set(xlabel='X', ylabel='Y', title='Scaled Moons')
        ax.legend(loc="upper right")
        plt.show()
        
def makecircle():                           #Yes, creating a circle inside a circle
    #random generator
    data,labels = ds.make_circles(n_samples=1000, shuffle=True,noise=0.05,random_state=None)
    
    #visualizing
    fig, ax = plt.subplots()
    ax.scatter(data[labels==0,0], data[labels==0,1], s=20,
               c="pink", label="pink") 
    ax.scatter(data[labels==1,0], data[labels==1,1], s=20,
               c="orange", label="orange")
    ax.set(xlabel="X",
           ylabel="Y",
           title="Circle")
    ax.legend(loc="upper right")
    plt.show()

def testingIPython():
    #Another pacakages needed
    from sklearn.datasets import make_gaussian_quantiles
    from sklearn.datasets import make_classification
    
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
    plt.subplot(321)
    plt.title("One informative feature, one cluster per class", fontsize='small')
    X1, Y1 = make_classification(n_features=2, n_redundant=0, 
                                 n_informative=1,n_clusters_per_class=1)
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
                s=25, edgecolor='k')
    
    plt.subplot(322)
    plt.title("Two informative features, one cluster per class",
              fontsize='small')
    X1, Y1 = make_classification(n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1)
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', 
                c=Y1, s=25, edgecolor='k')
    
    plt.subplot(323)
    plt.title("Two informative features, two clusters per class", fontsize='small')
    X2, Y2 = make_classification(n_features=2, n_redundant=0, n_informative=2)
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2, s=25, edgecolor='k')
    
    plt.subplot(324)
    plt.title("Multi-class, two informative features, one cluster", fontsize='small')
    X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2, 
                                 n_clusters_per_class=1, n_classes=3)
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, 
                s=25, edgecolor='k')
    
    plt.subplot(325)
    plt.title("Gaussian divided into three quantiles", fontsize='small')
    X1, Y1 = make_gaussian_quantiles(n_features=2, n_classes=3)
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')
    
    plt.show()
    
if __name__== "__main__":
    blob.randomcenter()
    blob.chosencenter()
    makemoon.normalmoon()
    makemoon.scaledmoon()
    makecircle()
    testingIPython()