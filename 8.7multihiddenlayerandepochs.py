#1
#Libraries
import pickle
from Modules.advanced_multihiddenlayerandepoch import NeuralNetwork
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
    epochs = 5
    #hidden layer [80,80]
    ANN = NeuralNetwork(network_structure=[image_pixels, 80, 80, no_of_different_labels]    #[Input, hidden1, hidd2n, output]
                        , learning_rate=0.01, bias=None)
    
    #train, yes. It's automatic fr fr 
    Weights=ANN.train(train_imgs, train_labels_one_hot, epochs=epochs, intermediate_results=True)
    
    #evaluate    
    for i in range(epochs):
        ANN.weights_matrices=Weights[i]
        print(f'epoch {i+1} :')
        correct, wrong=ANN.evaluate(train_imgs, train_labels)
        print(f'Evaluate Train: {correct/(correct+wrong)}')
        correct, wrong=ANN.evaluate(test_imgs, test_labels)
        print(f'Evaluate Test:  {correct/(correct+wrong)}')
        
    #run
    print("Run Test (n=20)")
    for i in range(20):
        z=ANN.run(test_imgs[i])
        print(f'{test_labels[i]} {z.argmax()}  {z.max()}')