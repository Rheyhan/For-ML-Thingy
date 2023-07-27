import numpy as np
from scipy.stats import truncnorm

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class NeuralNetwork:
    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate, bias=None):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.learning_rate = learning_rate
        self.bias = bias                                #bias=none (default)
        self.create_weight_matrices()
    
    def create_weight_matrices(self):
       """ A method to initialize the weight matrices of the neural 
       network with optional bias nodes"""
       bias_node = 1 if self.bias else 0
       
       #get weights in hidden
       rad = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
       X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
       self.wih = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes + bias_node))
        
       #get weights hidden out
       rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
       X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
       self.who = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes + bias_node))
    
    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can
        be tuple, list or ndarray
        """
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [self.bias]) )
            
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        
        output_vector1 = np.dot(self.wih, input_vector)
        output_hidden = activation_function(output_vector1)
        if self.bias:
            output_hidden = np.concatenate((output_hidden, [[self.bias]]))
        output_vector2 = np.dot(self.who, output_hidden)
        
        output_network = activation_function(output_vector2)
        output_errors = target_vector - output_network
        
        # update the weights:
        tmp = output_errors * output_network * (1.0 - output_network)
        tmp = self.learning_rate * np.dot(tmp, output_hidden.T)
        self.who += tmp
        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T,output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1,:]
        else:
            x = np.dot(tmp, input_vector.T)
        self.wih += self.learning_rate * x


    def run(self, input_vector):
        """
        input_vector can be tuple, list or ndarray
        """
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [1]))
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)
        
        if self.bias:
            output_vector = np.concatenate( (output_vector, [[1]]))
        output_vector = np.dot(self.who, output_vector)
        output_vector = activation_function(output_vector)
        return output_vector

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
    
    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm  
      
    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()
    
    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()