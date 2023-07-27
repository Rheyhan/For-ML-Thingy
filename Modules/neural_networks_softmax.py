import numpy as np
from scipy.stats import truncnorm

#random generator continous data normal
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

#logistic regression, making it in value of 0-1
@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)

#softmax Function
def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()
    
class NeuralNetwork:
    #init my beloved <3
    def __init__(self, no_of_in_nodes, no_of_out_nodes,
                 no_of_hidden_nodes, learning_rate, softmax=True):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.softmax=softmax
        self.create_weight_matrices()
         
    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural network"""
        #Get weight hidden in
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))
        
        #get weight hidden out
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))
        
    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can be tuples, lists or ndarrays
        """
        # make sure that the vectors have the right shape
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(input_vector.size, 1)
        output_vector_hidden = sigmoid(self.weights_in_hidden @ input_vector)
        
        #Get output error
        target_vector = np.array(target_vector).reshape(target_vector.size, 1)
        if self.softmax:
            output_vector_network = softmax(self.weights_hidden_out @ output_vector_hidden)
        else:
            output_vector_network = sigmoid(self.weights_hidden_out @ output_vector_hidden)
        output_error = target_vector - output_vector_network
        
        #get tmp and weights hidden out
        if self.softmax:
            ovn = output_vector_network.reshape(output_vector_network.size,)
            si_sj = - ovn * ovn.reshape(self.no_of_out_nodes, 1)
            s_der = np.diag(ovn) + si_sj
            tmp = s_der @ output_error
            self.weights_hidden_out += self.learning_rate * (tmp@ output_vector_hidden.T)
        else:
            tmp = output_error * output_vector_network * (1.0 - output_vector_network)
            self.weights_hidden_out += self.learning_rate * (tmp@ output_vector_hidden.T)
        
        # calculate hidden errors:
        hidden_errors = self.weights_hidden_out.T @ output_error
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        self.weights_in_hidden += self.learning_rate * (tmp @ input_vector.T)
    
    #It's just predict tbh
    def run(self, input_vector):
        """
        running the network with an input vector 'input_vector'.
        'input_vector' can be tuple, list or ndarray
        """
        # make sure that input_vector is a column vector:
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(input_vector.size, 1)
        input4hidden = sigmoid(self.weights_in_hidden @ input_vector)
        if self.softmax:
            output_vector_network = softmax(self.weights_hidden_out @ input4hidden)
        else:
            output_vector_network = sigmoid(self.weights_hidden_out @ input4hidden)
        return output_vector_network
            
    def evaluate(self, data, labels):
        """
        Counts how often the actual result corresponds to the
        target result.
        A result is considered to be correct, if the index of
        the maximal value corresponds to the index with the "1"
        in the one-hot representation,
        e.g.
        res = [0.1, 0.132, 0.875]
        labels[i] = [0, 0, 1]
        """
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs