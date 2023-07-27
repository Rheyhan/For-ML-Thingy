#libraries
import numpy as np
from scipy.stats import truncnorm
    #run/predict method
from scipy.special import expit as activation_function


#activation function!
def ReLU(x):
    return np.maximum(0.0, x)

def ReLU_derivation(x):
    if x <= 0:
        return 0
    else:
        return 1
    
    
#data distribution/generator
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

class NeuralNetwork:
    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()
         
    def create_weight_matrices(self):
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,
                                        self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, 
                                         self.no_of_hidden_nodes))
        
    def train(self):
        pass
    
    #It's just predict tbh
    def run(self, input_vector):
        '''
        running the network with an input vector 'input_vector'.
        'input_vector' can be tuple, list or ndarray
        '''
        # turning the input vector into a column vector
        input_vector = np.array(input_vector, ndmin=2).T
        input_hidden = activation_function(self.weights_in_hidden @ input_vector)
        output_vector = activation_function(self.weights_hidden_out @ input_hidden)
        return output_vector
 
if __name__ == "__main__":
    simple_network = NeuralNetwork(no_of_in_nodes=2,
                                   no_of_out_nodes=2,
                                   no_of_hidden_nodes=4,
                                   learning_rate=0.6)      #configurable
    
    print(simple_network.weights_in_hidden)     #check weights tertutup
    print(simple_network.weights_hidden_out)    #check weights terbuka
    
    #run
    print(simple_network.run([(3, 4)]))
