'''
Create a
Neural Network to classify the 'flowers'
'''

import pandas as pd
import numpy as np
from Modules.neural_networks2 import NeuralNetwork
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    #input data
    df=np.loadtxt("Files/Flowerthingy.txt", delimiter=" ")
    data=df[:,:-1]                  #get main data
    n_classes=data.shape[1]         #get var n
    labels=df[:,-1]                 #get label

    #convert categoric -> multiple array coll
    labels = np.arange(n_classes) == labels.reshape(labels.size, 1)
    labels = labels.astype(float)
    
    #scaling data
    data=scale(data)
    
    #split train test (train 0.8)
    train_data, test_data, train_labels, test_labels= train_test_split(data, labels,  train_size=0.8, random_state=375437)
    
    #get neutral network model (No Bias)
    simple_network = NeuralNetwork(no_of_in_nodes=4, no_of_out_nodes=4, no_of_hidden_nodes=20, learning_rate=0.3)
    
    #train
    for i in range(len(train_data)):
        simple_network.train(train_data[i], train_labels[i])
    
    #evaluate
    print(simple_network.evaluate(train_data, train_labels))