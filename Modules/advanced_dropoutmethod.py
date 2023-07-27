import numpy as np
import random
from scipy.special import expit as activation_function
from scipy.stats import truncnorm
import sys
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def progressBar(count_value, total, suffix=''):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()
    
class NeuralNetwork:
    def __init__(self, no_of_in_nodes, no_of_out_nodes, 
                 no_of_hidden_nodes, learning_rate, bias=None):  
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes       
        self.no_of_hidden_nodes = no_of_hidden_nodes          
        self.learning_rate = learning_rate 
        self.bias = bias
        self.create_weight_matrices()
        
    def create_weight_matrices(self):
        bias_node = 1 if self.bias else 0

        n = (self.no_of_in_nodes + bias_node) * self.no_of_hidden_nodes
        X = truncated_normal(mean=2, sd=1, low=-0.5, upp=0.5)
        self.wih = X.rvs(n).reshape((self.no_of_hidden_nodes,  self.no_of_in_nodes + bias_node))

        n = (self.no_of_hidden_nodes + bias_node) * self.no_of_out_nodes
        X = truncated_normal(mean=2, sd=1, low=-0.5, upp=0.5)
        self.who = X.rvs(n).reshape((self.no_of_out_nodes, (self.no_of_hidden_nodes + bias_node)))

    def dropout_weight_matrices(self, active_input_percentage=0.70,
                                active_hidden_percentage=0.70):
        # restore wih array, if it had been used for dropout
        self.wih_orig = self.wih.copy()
        self.no_of_in_nodes_orig = self.no_of_in_nodes
        self.no_of_hidden_nodes_orig = self.no_of_hidden_nodes
        self.who_orig = self.who.copy()
        
        active_input_nodes = int(self.no_of_in_nodes * active_input_percentage)
        active_input_indices = sorted(random.sample(range(0, self.no_of_in_nodes), 
                                      active_input_nodes))
        
        active_hidden_nodes = int(self.no_of_hidden_nodes * active_hidden_percentage)
        active_hidden_indices = sorted(random.sample(range(0, self.no_of_hidden_nodes), 
                                       active_hidden_nodes))
        
        self.wih = self.wih[:, active_input_indices][active_hidden_indices]       
        self.who = self.who[:, active_hidden_indices]
        
        self.no_of_hidden_nodes = active_hidden_nodes
        self.no_of_in_nodes = active_input_nodes
        
        return active_input_indices, active_hidden_indices
    
    def weight_matrices_reset(self, 
                              active_input_indices, 
                              active_hidden_indices):
        
        """
        self.wih and self.who contain the newly adapted values from the active nodes.
        We have to reconstruct the original weight matrices by assigning the new values 
        from the active nodes
        """
 
        temp = self.wih_orig.copy()[:,active_input_indices]
        temp[active_hidden_indices] = self.wih
        self.wih_orig[:, active_input_indices] = temp
        self.wih = self.wih_orig.copy()

        self.who_orig[:, active_hidden_indices] = self.who
        self.who = self.who_orig.copy()
        self.no_of_in_nodes = self.no_of_in_nodes_orig
        self.no_of_hidden_nodes = self.no_of_hidden_nodes_orig
 
           
    
    def train_single(self, input_vector, target_vector):
        """ 
        input_vector and target_vector can be tuple, list or ndarray
        """
 
        if self.bias:
            # adding bias node to the end of the input_vector
            input_vector = np.concatenate( (input_vector, [self.bias]) )

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.wih, input_vector)
        output_vector_hidden = activation_function(output_vector1)
        
        if self.bias:
            output_vector_hidden = np.concatenate( (output_vector_hidden, [[self.bias]]) )
        
        output_vector2 = np.dot(self.who, output_vector_hidden)
        output_vector_network = activation_function(output_vector2)
        
        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)     
        tmp = self.learning_rate  * np.dot(tmp, output_vector_hidden.T)
        self.who += tmp


        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1,:] 
        else:
            x = np.dot(tmp, input_vector.T)
        self.wih += self.learning_rate * x

            
        
    def train(self, data_array, 
              labels_one_hot_array,
              epochs=1,
              active_input_percentage=0.70,
              active_hidden_percentage=0.70,
              no_of_dropout_tests = 10):

        partition_length = int(len(data_array) / no_of_dropout_tests)
        
        for epoch in range(epochs):
            progressBar(epoch, epochs)
            for start in range(0, len(data_array), partition_length):
                active_in_indices, active_hidden_indices = \
                           self.dropout_weight_matrices(active_input_percentage,
                                                        active_hidden_percentage)
                for i in range(start, start + partition_length):
                    self.train_single(data_array[i][active_in_indices], 
                                     labels_one_hot_array[i]) 
                    
                self.weight_matrices_reset(active_in_indices, active_hidden_indices)
        print("Train's finished")    
    
    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
        
        if self.bias:
            # adding bias node to the end of the input_vector
            input_vector = np.concatenate( (input_vector, [self.bias]) )
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)
        
        if self.bias:
            output_vector = np.concatenate( (output_vector, [[self.bias]]) )
            

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
    
    def confusion_matrix(self, data, labels):
        cm = {}
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            target = labels[i][0]
            if (target, res_max) in cm:
                cm[(target, res_max)] += 1
            else:
                cm[(target, res_max)] = 1
        return cm