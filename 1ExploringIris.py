'''Visualizing Iris'''

#libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
#load iris
iris = load_iris()


colors = ['blue', 'red', 'green']

def histogram():
    #histogram of the features
    fig, ax = plt.subplots()
    x_index = 3
    for label, color in zip(range(len(iris.target_names)), colors):
        ax.hist(iris.data[iris.target==label, x_index], 
                label=iris.target_names[label], 
                color=color)
    ax.set_xlabel(iris.feature_names[x_index])
    ax.legend(loc='upper right')
    fig.show()
    plt.show()
    
def scatter():
    #scatterplot with two features
    fig, ax = plt.subplots()
    x_index = 3
    y_index = 0
    for label, color in zip(range(len(iris.target_names)), colors):
        ax.scatter(iris.data[iris.target==label, x_index],
                iris.data[iris.target==label, y_index],
                label=iris.target_names[label],
                c=color)
    ax.set_xlabel(iris.feature_names[x_index])
    ax.set_ylabel(iris.feature_names[y_index])
    ax.legend(loc='upper left')
    plt.show()


def matrixplotmanual():
    #scatterplot dari semua peubah
    n = len(iris.feature_names)
    fig, ax = plt.subplots(n, n, figsize=(16, 16))
    for x in range(n):
        for y in range(n):
            xname = iris.feature_names[x]
            yname = iris.feature_names[y]
            for color_ind in range(len(iris.target_names)):
                ax[x, y].scatter(iris.data[iris.target==color_ind,x],
                                 iris.data[iris.target==color_ind, y],
                                 label=iris.target_names[color_ind],
                                 c=colors[color_ind])
            ax[x, y].set_xlabel(xname)
            ax[x, y].set_ylabel(yname)
            ax[x, y].legend(loc='upper left')
    plt.show()

def matrixplototomatis():
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    pd.plotting.scatter_matrix(iris_df,
                               c=iris.target,figsize=(8, 8)
                               )
    plt.show()

def threeDVisual():
    X=[]
    for iclass in range(3):
        X.append([[],[],[]])
        for i in range(len(iris.data)):
            if (iris.target[i]==iclass):
                X[iclass][0].append(iris.data[i][0])
                X[iclass][1].append(iris.data[i][1])
                X[iclass][2].append(sum(iris.data[i][2:]))
        
    colours=("r","g","y")
    fig=plt.figure()
    ax=fig.add_subplot(111,projection="3d")
        
    for iclass in range(3):
        ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colours[iclass])
    plt.show()
        
if __name__ == "__main__":
    histogram()
    scatter()
    
    matrixplotmanual()
    matrixplototomatis()
    
    threeDVisual()