import numpy as np
import pickle
from Modules.advanced_dropoutmethod import NeuralNetwork


with open("Files/pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)
train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

partition_length = int(len(train_imgs) / no_of_different_labels)
print(partition_length)

start = 0
for start in range(0, len(train_imgs), partition_length):
    print(start, start + partition_length)
epochs = 3

simple_network = NeuralNetwork(no_of_in_nodes = image_pixels, 
                               no_of_out_nodes = 10, 
                               no_of_hidden_nodes = 100,
                               learning_rate = 0.1)
    
simple_network.train(train_imgs, 
                     train_labels_one_hot, 
                     active_input_percentage=1,
                     active_hidden_percentage=1,
                     no_of_dropout_tests = 100,
                     epochs=epochs)

#evaluate
corrects, wrongs = simple_network.evaluate(train_imgs, train_labels)
print(f'accuracy train: {corrects / ( corrects + wrongs)}   |Corrects: {corrects}    |Wrongs: {wrongs}')
corrects, wrongs = simple_network.evaluate(test_imgs, test_labels)
print(f'accuracy train: {corrects / ( corrects + wrongs)}   |Corrects: {corrects}    |Wrongs: {wrongs}')

#run and its original value
for i in range(20):
    z=simple_network.run(test_imgs[i])
    print(f'{test_labels[i]}    {z.argmax()}    {z.max()}')

#confusion matrix
cm=simple_network.confusion_matrix(train_imgs, train_labels)
cm = list(cm.items())
print(sorted(cm))