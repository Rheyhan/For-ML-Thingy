'''
Simple Neural Networks
#Just a glorified linear line cuh
dsxz
(x1 + x2) -> [Perceptron] -> Output
Sc:
Bernd_Klein ML
-https://towardsdatascience.com/deep-learning-with-python-neural-networks-complete-tutorial-6b53c0b06af0
'''

#libraries needed
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from Modules.Perceptrons import Perceptron
#dummy data
input1=[0,0,1,1]
input2=[0,1,0,1]
output=[0,0,0,1]

class singleperceptron:
    def Normal():
        fig, ax=plt.subplots()
        xmin, xmax = -0.2, 1.4
        X = np.arange(xmin, xmax, 0.1)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([-0.1, 1.1])    
        for i,j,k in zip(input1, input2, output):
            if (k==1):
                ax.scatter(i,j,color="g")    
            else:
                ax.scatter(i,j,color="r")
        m, c = -1, 1.2
        ax.plot(X, m * X + c )
        plt.show()
        
    def WithBias():
        def labelled_samples(n):
           for _ in range(n):
                s = np.random.randint(0, 2, (2,))
                yield (s, 1) if s[0] == 1 and s[1] == 1 else (s, 0)
       
        p = Perceptron(weights=[0.3, 0.3, 0.3], learning_rate=0.2)
       
        for in_data, label in labelled_samples(30):
            p.adjust(label, in_data)
        test_data, test_labels = list(zip(*labelled_samples(30)))
        
        evaluation = p.evaluate(test_data, test_labels)
        print(evaluation)

        fig, ax=plt.subplots()
        xmin, xmax = -0.2, 1.4
        X = np.arange(xmin, xmax, 0.1)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([-0.1, 1.1])    
        for i,j,k in zip(input1, input2, output):
            if (k==1):
                ax.scatter(i,j,color="g")    
            else:
                ax.scatter(i,j,color="r")
                
        m = -p.weights[0] / p.weights[1]
        c = -p.weights[2] / p.weights[1]
        print(m, c)
        ax.plot(X, m * X + c )
        plt.show()
            
if __name__== "__main__":
    singleperceptron.Normal()
    singleperceptron.WithBias()
    
    

#=======================================Garis Suci============================================ 
"""EXERCISE"""
'''
1.
Input1      Input 2     Output
x1 < 0.5    x2 < 0.5    0
x1 < 0.5    x2 >= 0.5   0
x1 >= 0.5   x2 < 0.5    0
x1 >= 0.5   x2 >= 0.5   1
Why neural network with one perceptron isn't gonna work?

2.
Make a neuyral network with one perceptron, if x1 < 0.5 belongs to class 0. If x1>= 0.5 belongs to class 1 
'''

#libraries
import numpy as np
from Modules.Perceptrons import Perceptron
import matplotlib.pyplot as plt

def firstanswer():
    p = Perceptron(weights=[0.3, 0.3, 0.3], bias=1,
                   learning_rate=0.2)
    
    def labelled_samples(n):
        for _ in range(n):
            s = np.random.random(2) #2 random float 0-1
            yield (s, 1) if (s[0] >= 0.5 and s[1] >= 0.5) else (s, 0)   #if x1 and x2 >= 0.5 maka yield 1
    
    for in_data, label in labelled_samples(200):
        p.adjust(label, in_data)

    test_data, test_labels = list(zip(*labelled_samples(300)))       # n=300
    
    evaluation = p.evaluate(test_data, test_labels)
    print(evaluation)               #Evaluasi Model


    ones = [test_data[i] for i in range(len(test_data)) if test_labels[i] == 1]
    zeroes = [test_data[i] for i in range(len(test_data)) if test_labels[i] == 0]
    
    fig, ax = plt.subplots()
    xmin, xmax = -0.2, 1.2
    
    #plot label=1
    X, Y = list(zip(*ones))
    ax.scatter(X, Y, color="g")
    
    #plot label =0
    X, Y = list(zip(*zeroes))
    ax.scatter(X, Y, color="r")
    
    #set x and y limit
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([-0.1, 1.1])
    
    #the funni seperatable line
    c = -p.weights[2] / p.weights[1]
    m = -p.weights[0] / p.weights[1]
    X = np.arange(xmin, xmax, 0.1)
    
    ax.plot(X, m * X + c, label="decision boundary")
    plt.show()
    
def secondanswer():
    p = Perceptron(weights=[0.3, 0.3, 0.3], bias=1,
                   learning_rate=0.2)
    
    def labelled_samples(n):
        for _ in range(n):
            s = np.random.random(2) #2 random float 0-1
            yield (s, 1) if (s[0] >= 0.5) else (s, 0)   #if x1 >= 0.5 maka yield 1
    
    for in_data, label in labelled_samples(400):
        p.adjust(label, in_data)

    test_data, test_labels = list(zip(*labelled_samples(600)))       # n=600
    
    evaluation = p.evaluate(test_data, test_labels)
    print(evaluation)               #Evaluasi Model


    ones = [test_data[i] for i in range(len(test_data)) if test_labels[i] == 1]
    zeroes = [test_data[i] for i in range(len(test_data)) if test_labels[i] == 0]
    
    fig, ax = plt.subplots()
    xmin, xmax = -0.2, 1.2
    
    #plot label=1
    X, Y = list(zip(*ones))
    ax.scatter(X, Y, color="g")
    
    #plot label =0
    X, Y = list(zip(*zeroes))
    ax.scatter(X, Y, color="r")
    
    #set x and y limit
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([-0.1, 1.1])
    
    #the funni seperatable line
    c = -p.weights[2] / p.weights[1]
    m = -p.weights[0] / p.weights[1]
    X = np.arange(xmin, xmax, 0.1)
    
    ax.plot(X, m * X + c, label="decision boundary")
    plt.show()
    
    
if __name__ =="__main__":
    firstanswer()
    #Nah, can't use seperatable line between two categories in here mate, the straight linear line doesn't work!
    #stick with a curved line instead!
    secondanswer()