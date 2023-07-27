'''
Self explanotary
we're making it out of the hood with this one 🗣️🗣️🔥🔥💥💥
'''

#1
#Libraries
import pickle
from Modules.advanced_epochandbias import NeuralNetwork
import numpy as np

#load data
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
    epochs=4
    #get model
    ANN = NeuralNetwork(image_pixels, no_of_different_labels, 100, 0.15, None)
    #train
    weights = ANN.train(train_imgs, train_labels_one_hot, 
                            epochs=epochs, intermediate_results=True)
    
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
        print("accuracy test: ", corrects / ( corrects + wrongs))