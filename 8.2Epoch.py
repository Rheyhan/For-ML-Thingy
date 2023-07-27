'''
We use a method named "epoch" to initiate a multiple train
'''

#1
import pickle
import numpy as np
from Modules.advanced import NeuralNetwork
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

def manualepoch(epochs):
    '''Epoch must be int'''
    #multiple train!    #this sh takes so long to run :sob:
    for epoch in range(epochs):
        for i in range(len(train_imgs)):
            ANN.train(train_imgs[i], train_labels_one_hot[i])
        print(f'epoch   {epoch}')
        corrects, wrongs = ANN.evaluate(train_imgs, train_labels)
        print("accuracy train: ", corrects / ( corrects + wrongs))
        corrects, wrongs = ANN.evaluate(test_imgs, test_labels)
        print("accuracy: test", corrects / ( corrects + wrongs))    

if __name__ =="__main__":
    ANN= NeuralNetwork(image_pixels, 10, 100, 0.1)
    manualepoch(3)
    
    
    
    
    
    
    
    
    
    
    
#2 With a bulit in module
import pickle
import numpy as np
from Modules.advanced_epoch import NeuralNetwork
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
    ANN = NeuralNetwork(no_of_in_nodes = image_pixels, no_of_out_nodes = 10,
                        no_of_hidden_nodes = 100, learning_rate = 0.15)
    
    #THIS PIECE OF SH TAKES SO LONG TO TRAIN    (opech=3)
    epochs=3
    weights = ANN.train(train_imgs, 
                        train_labels_one_hot, epochs,   
                        intermediate_results=True)
    
    #predicting things
    for i in range(20):
        res = ANN.run(test_imgs[i])
        print(test_labels[i], np.argmax(res), np.max(res))
    
    #confusion matrix
    cm = ANN.confusion_matrix(train_imgs, train_labels)
    cm = list(cm.items())
    print(sorted(cm))
    
    #evaluate!
    for i in range(epochs):
        print("epoch: ", i+1)
        ANN.wih = weights[i][0]
        ANN.who = weights[i][1]
        corrects, wrongs = ANN.evaluate(train_imgs, train_labels)
        print("accuracy train: ", corrects / ( corrects + wrongs))
        corrects, wrongs = ANN.evaluate(test_imgs, test_labels)
        print("accuracy: test", corrects / ( corrects + wrongs))