'''
You may have noticed that it is quite slow to read in the data from the csv files.
We will save the data in binary format with the dump function from the pickle module:
'''

''' Convert csv-> pkl (Not needed actually)
import pandas as pd
df=pd.read_csv("mnist_test.csv")
df.to_pickle('mnist_test.pkl') 
df=pd.read_csv("mnist_train.csv")
df.to_pickle('mnist_train.pkl')
'''

#ibraries
import pickle
from Modules.advanced import NeuralNetwork
import numpy as np

#load data
import pickle
with open("Files/pickled_mnist.pkl", "br") as fh:  
    data = pickle.load(fh)
    
train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]

no_of_different_labels = 10             #(0-9), there's 10
image_pixels = 28**2                    #pic resolution


if __name__ == "__main__":
    #get model
    ANN = NeuralNetwork(no_of_in_nodes = image_pixels,  #pic res
                        no_of_out_nodes = 10,           #10 labels ony
                        no_of_hidden_nodes = 100,
                        learning_rate = 0.1)
    
    #train
    for i in range(len(train_imgs)):
        ANN.train(train_imgs[i], train_labels_one_hot[i])
        
    #predicting
    for i in range(20):
        res = ANN.run(test_imgs[i])
        print(test_labels[i], np.argmax(res), np.max(res))
        
    #evaluate the model based from the train and test!
        #check train
    corrects, wrongs = ANN.evaluate(train_imgs, train_labels)
    print("accuracy train: ", corrects / ( corrects + wrongs))
        #check test
    corrects, wrongs = ANN.evaluate(test_imgs, test_labels)
    print("accuracy: test", corrects / ( corrects + wrongs))
    
        #if you wanna usee confusion matrix cuz why not?
    cm = ANN.confusion_matrix(train_imgs, train_labels)
    print(cm)
        
        #check each values with precision and recall method
    for i in range(10):
        print("digit: ", i, "precision: ", ANN.precision(i, cm), "recall: ", ANN.recall(i, cm))
    