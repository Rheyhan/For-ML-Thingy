#libraries
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Modules.neural_networks1 import NeuralNetwork

if __name__ == "__main__":
    #making a random data (blob methods)
    n=500
    blob_centers=([2, 6], [6, 2], [7, 7])
    k=len(blob_centers)
    data, labels=make_blobs(n, centers=blob_centers,  random_state=12873)
    
    #visualizing
    colours=("pink", "red", "green")
    fig, ax = plt.subplots()
    
    for i in range(k):
        ax.scatter(data[labels==i][:,0], 
                   data[labels==i][:,1],
                   s=20, 
                   c=colours[i], label=str(i))
    plt.show()
    
    #split
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels,
                                                                        train_size=0.8,random_state=42)
    
        #change labels representation
    train_labels_one_hot = np.arange(k) == train_labels.reshape(train_labels.size, 1)
    train_labels_one_hot = train_labels_one_hot.astype(float)
    
    test_labels_one_hot = np.arange(k) == test_labels.reshape(test_labels.size, 1)
    test_labels_one_hot = test_labels_one_hot.astype(float)
    
    #creating neural network
    simple_network = NeuralNetwork(no_of_in_nodes=2, no_of_out_nodes=3, 
                                   no_of_hidden_nodes=5, learning_rate=0.3)
    
    #train network with training data
    for i in range(len(train_data)):
        simple_network.train(train_data[i], train_labels_one_hot[i])
    print(simple_network.evaluate(train_data, train_labels_one_hot))
    
    #Visualization of evaluate
    for i in range(len(train_data)):
        z=(simple_network.run(train_data[i])).argmax()
        if z==train_labels[i]:
            plt.scatter(train_data[i][0],train_data[i][1], c="green", s=20)
        else:
            plt.scatter(train_data[i][0],train_data[i][1], c="red", s=20)
    plt.show()