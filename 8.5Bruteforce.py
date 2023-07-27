#1  Brute force to get best evaluation, saved in csv format
import pickle
from Modules.advanced_epochandbias import NeuralNetwork
if __name__ == "__main__":
    with open("pickled_mnist.pkl", "br") as fh:  
        data = pickle.load(fh)
    train_imgs = data[0]
    test_imgs = data[1]
    train_labels = data[2]
    test_labels = data[3]
    train_labels_one_hot = data[4]
    test_labels_one_hot = data[5]
    
    epochs = 12
    no_of_different_labels = 10
    image_pixels = 28**2 
    with open("initest.csv", "w") as fh_out:
        for hidden_nodes in [20, 50, 100, 120, 150]:
            for learning_rate in [0.01, 0.05, 0.1, 0.2]:
                for bias in [None, 0.5]:
                    network = NeuralNetwork(no_of_in_nodes=image_pixels, no_of_out_nodes=10,
                                            no_of_hidden_nodes=hidden_nodes,
                                            learning_rate=learning_rate,
                                            bias=bias)
                    weights = network.train(train_imgs, train_labels_one_hot, epochs=epochs, intermediate_results=True)
                    for epoch in range(epochs):
                        print("*", end="")
                        network.wih = weights[epoch][0]
                        network.who = weights[epoch][1]
                        train_corrects, train_wrongs = network.evaluate(train_imgs, train_labels)
                        test_corrects, test_wrongs = network.evaluate(test_imgs,test_labels)
                        outstr = str(hidden_nodes) + " " + str(learning_rate) + " " + str(bias)
                        outstr += " " + str(epoch) + " "
                        outstr += str(train_corrects / (train_corrects + train_wrongs)) + " "
                        outstr += str(train_wrongs / (train_corrects + train_wrongs)) + " "
                        outstr += str(test_corrects / (test_corrects + test_wrongs)) + " "
                        outstr += str(test_wrongs / (test_corrects + test_wrongs))
                        fh_out.write(outstr + "\n" )
                        fh_out.flush()
                    print("\n")
                    
                    
                    
                    
                    
                    

from Modules.neural_networks_Bruteforce import brute
import pickle

if __name__ == "__main__":
    with open("Files\pickled_mnist.pkl", "br") as fh:  
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
    path="Pickled_mnist_Brute.csv"
    test=brute(image_pixels, no_of_different_labels, train_imgs, train_labels, test_imgs, test_labels, train_labels_one_hot)
    test.start(path, hidden_nodes=[20,50,100,150,200], learning_rates=[0.01, 0.05, 0.1], biases=[None, 0.05], epochs=4)