'''
The
softmax function, also known as softargmax or normalized
exponential function, is a function that takes as input a vector of
n real numbers, and normalizes it into a probability distribution
consisting of n probabilities proportional to the exponentials of
the input vector. A probability distribution implies that the result
vector sums up to 1. Needless to say, if some components of the
input vector are negative or greater than one, they will be in the
range (0, 1) after applying Softmax . The Softmax function is
often used in neural networks, to map the results of the output
layer, which is non-normalized, to a probability distribution over
predicted output classes.
'''

from sklearn.datasets import make_blobs
from Modules.neural_networks_softmax import NeuralNetwork
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter

if __name__ == "__main__":
    #random blob generator (I love blob <3)
    n=300
    thisiscenter=([2,6], [6,2])
    k=len(thisiscenter)
    samples, labels = make_blobs(n, centers=thisiscenter, random_state=0)
    
    #
    colours=("black", "red", "cyan", "pink", "purple")
    fig, ax= plt.subplots()
    for i in range(k):
        ax.scatter(samples[labels==i][:,0],
                   samples[labels==i][:,1],
                   s=20,c=colours[i], label=str(i))
    plt.show()
    
    TheNetwork= NeuralNetwork(k, k, 5, 0.3, True)
        
    #manual train test split because why not?
    size_of_learn_sample = int(n*0.8)
    learn_data = samples[:size_of_learn_sample]
    test_data = samples[-size_of_learn_sample:]
    labels_one_hot = (np.arange(2) == labels.reshape(labels.size, 1))
    labels_one_hot = labels_one_hot.astype(float)
    
    #train the data
    for i in range(size_of_learn_sample):
        #print(learn_data[i], labels[i], labels_one_hot[i])
        TheNetwork.train(learn_data[i], labels_one_hot[i])
        
    evaluation = Counter()
    print(TheNetwork.evaluate(learn_data, labels))
    
    #run
    for x in [(1, 4), (2, 6), (3, 3), (6, 2)]:          #an Example
        y = TheNetwork.run(x)
        print(x, y.argmax())