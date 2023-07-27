import numpy as np
from scipy.stats import truncnorm
from scipy.special import expit as activation_function

#random generator continous data
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
       self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes + bias_node))
        
       #get weights hidden out
       rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
       X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
       self.weights_hidden_out = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes + bias_node))
    
    def train(self, input_vector, target_vector):
        """ input_vector and target_vector can be tuple, list or ndarray """
        # make sure that the vectors have the right shap
        
        #input vector reshaping
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(input_vector.size,1)
        
        if self.bias:
        # adding bias node to the end of the input_vector
            input_vector = np.concatenate((input_vector, [[self.bias]]))
            
        target_vector = np.array(target_vector).reshape(target_vector.size, 1)
        
        output_vector_hidden = activation_function(self.weights_in_hidden @ input_vector)
        if self.bias:
            output_vector_hidden = np.concatenate((output_vector_hidden, [[self.bias]]))

        output_vector_network = activation_function(self.weights_hidden_out @ output_vector_hidden)
        output_error = target_vector - output_vector_network
        
        # update the weights:
        tmp = output_error * output_vector_network * (1.0 - output_vector_network)
        self.weights_hidden_out += self.learning_rate * (tmp @ output_vector_hidden.T)
        
        # calculate hidden errors:
        hidden_errors = self.weights_hidden_out.T @ output_error
        
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        if self.bias:
            x = (tmp @input_vector.T)[:-1,:] # last row cut of f,
        else:
            x = tmp @ input_vector.T
        self.weights_in_hidden += self.learning_rate * x

    def run(self, input_vector):
        """
        running the network with an input vector 'input_vector'.
        'input_vector' can be tuple, list or ndarray
        """
        # make sure that input_vector is a column vector:
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(input_vector.size, 1)
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate( (input_vector, [[1]]))
            
        input4hidden = activation_function(self.weights_in_hidden@input_vector)
        if self.bias:
            input4hidden = np.concatenate( (input4hidden, [[1]]) )
        output_vector_network = activation_function(self.weights_hidden_out @ input4hidden)
        return output_vector_network

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i].argmax():
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs