'''
Dropping out
can be seen as temporarily deactivating or ignoring
neurons of the network. This technique is applied in
the training phase to reduce overfitting effects.
Overfitting is an error which occurs when a network
is too closely fit to a limited set of input samples
'''
#libraries
import numpy as np
import random as rd


if __name__ == "__main__":
    #(10nodes)->(5nodes)->(4nodes)
    input_nodes=10
    hidden_nodes=5
    output_nodes=4
    
    #get wih with rand
    wih=np.random.randint(-10, 10, (hidden_nodes, input_nodes))
    print(wih)
    
    #get active_input_indices
    active_input_percentage=0.7
    active_input_nodes=int(active_input_percentage*input_nodes)
    active_input_indices = sorted(rd.sample(range(0, input_nodes),active_input_nodes))
    print(active_input_indices)
    
    #wih after deactivating input nodes
    wih_old = wih.copy()
    wih = wih[:, active_input_indices]
    print(wih,"\n")
    
    #get who with rand
    who = np.random.randint(-5, 5, (output_nodes, hidden_nodes))
    print(who)
    
    #get activate_hidden_indices
    active_hidden_percentage = 0.7
    active_hidden_nodes = int(hidden_nodes * active_hidden_percentage)
    active_hidden_indices = sorted(rd.sample(range(0, hidden_nodes),active_hidden_nodes))
    print(active_hidden_indices)
    
    #who after deactivating hidden indices
    who_old = who.copy()
    who = who[:, active_hidden_indices]
    print(who,"\n")
    
    #wih after deactivating hidden indices (must deactivating input beforehand)
    wih = wih[active_hidden_indices]
    print(wih)