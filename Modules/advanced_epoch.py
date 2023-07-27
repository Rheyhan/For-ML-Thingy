import numpy as np
from scipy.stats import truncnorm
import sys
import time

def progressBar(count_value, total, suffix=''):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()
    
@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class NeuralNetwork:
    def __init__(self, no_of_in_nodes, no_of_out_nodes,no_of_hidden_nodes, learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """
        A method to initialize the weight
        matrices of the neural network
        """
        #get wih
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes)) 
        
        #get who
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.who = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))

    def train_single(self, input_vector, target_vector):
        """
        input_vector and target_vector can
        be tuple, list or ndarray
        """
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        output_vector1 = np.dot(self.wih, input_vector)
        output_hidden = activation_function(output_vector1)
        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)
        output_errors = target_vector - output_network
        
        # update the weights:
        tmp = output_errors * output_network * (1.0 - output_network)
        tmp = self.learning_rate * np.dot(tmp, output_hidden.T)
        self.who += tmp
        
        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # update the weights:   
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        self.wih += self.learning_rate \
        * np.dot(tmp, input_vector.T)
    
    def train(self, data_array, labels_one_hot_array,
              epochs=1, intermediate_results=False):
        intermediate_weights = []
        for epoch in range(epochs):
            progressBar(epoch, epochs)
            for i in range(len(data_array)):
                self.train_single(data_array[i], labels_one_hot_array[i])
            if intermediate_results:
                intermediate_weights.append((self.wih.copy(), self.who.copy()))
        print("Train completed")
        return intermediate_weights

    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)
        output_vector = np.dot(self.who, output_vector)
        output_vector = activation_function(output_vector)
        return output_vector
    
    def confusion_matrix(self, data_array, labels):
        cm = {}
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            if (target, res_max) in cm:
                cm[(target, res_max)] += 1
            else:
                cm[(target, res_max)] = 1
        return cm
      
    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()
    
    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()
    
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
               corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs