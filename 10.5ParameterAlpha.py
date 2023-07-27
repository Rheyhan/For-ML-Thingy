'''
A comparison of different values for regularization parameter ‘alpha’ on synthetic datasets. The plot shows
that different alphas yield different decision functions.

Alpha is a parameter for regularization term, aka penalty term, that combats overfitting by constraining the
size of the weights. Increasing alpha may fix high variance (a sign of overfitting) by encouraging smaller
weights, resulting in a decision boundary plot that appears with lesser curvatures. Similarly, decreasing alpha
may fix high bias (a sign of underfitting) by encouraging larger weights, potentially resulting in a more
complicated decision boundary.
'''

#libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
import sys

def progressBar(count_value, total, suffix=''):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()
    
if __name__ == "__main__":
    alphas=np.logspace(-1,1,5)              #alphas gonna be used
    
    classifiers=[]
    names=[]
    
    for alpha in alphas:
        classifiers.append(make_pipeline(StandardScaler(), 
                                         MLPClassifier(solver="lbfgs", alpha=alpha, random_state=1232, max_iter=217821, early_stopping=True, hidden_layer_sizes=[100,100],)
                                         )
                           )
        names.append(f'alpha {alpha:.2f}')
        
    #Dummy data generator (2 X variables, and y categoric)
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=378237, n_clusters_per_class=1)
    
    rng = np.random.RandomState(2)          #setseed
    X += 2 * rng.uniform(size=X.shape)      #randomizing x with uniform
    linearly_separable = (X, y)

    datasets = [make_moons(noise=0.3, random_state=0),                  #make moons
                make_circles(noise=0.2, factor=0.5, random_state=1),    #make circles
                linearly_separable]                                     #data from earlier
    #Each data on datasets gonna have the same size
    print(len(datasets))
    
    #Visualize plot!
    h = .02 # step size in the mesh
    figure = plt.figure(figsize=(17,9)) #17x9 size
    
    progresscounter=0;i=1
    for data, labels in datasets:
        progressBar(progresscounter, len(datasets))
        
        Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=.3)
        
        #for x and y border size 
        x_min, x_max = data[:, 0].min()-0.5, data[:, 0].max()+0.5
        y_min, y_max = data[:, 1].min()-0.5, data[:, 1].max()+0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        #colour choices and how many plots there will be
        cm = plt.cm.RdBu                #RdBu Colormap
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        
        # Plot the training points
        ax.scatter(Data_train[:, 0], Data_train[:, 1], c=Labels_train, cmap=cm_bright)
        
        # and testing points
        ax.scatter(Data_test[:, 0], Data_test[:, 1], c=Labels_test, cmap=cm_bright, alpha=0.6)
        
        #limit the view point of the plot
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
        
        #plot for each different classifiers rahh
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(Data_train, Labels_train)
            score=clf.score(Data_test, Labels_test)     #Get score of the test
            
            #draw color plot    
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            Z = Z.reshape(xx.shape)
                
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
        
            # Plot also the training points
            ax.scatter(Data_train[:, 0], Data_train[:, 1], c=Labels_train, cmap=cm_bright,
                    edgecolors='black', s=25)
            # and testing points
            ax.scatter(Data_test[:, 0], Data_test[:, 1], c=Labels_test, cmap=cm_bright,
                    alpha=0.6, edgecolors='black', s=25)
            
            #limit the view point of the plot
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            
            ax.set_title(name)
            
            #Putting a text of the score
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip("0"),
                    size=15, horizontalalignment='right')
            i += 1
        progresscounter+=1
    figure.subplots_adjust(left=.02, right=.98)
    print("Tasl finished successfully")
    plt.show()