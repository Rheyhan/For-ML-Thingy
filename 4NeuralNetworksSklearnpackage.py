'''
Neural Network but with a SKLEAN package
'''

#libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import random

if __name__ == "__main__":
    #dummies
    n=50
    data, labels = make_blobs(n, centers=([1.2, 3], [4.5, 6.9]), random_state=23567)
    colours=("green", "orange")

    #visualize blobs
    fig, ax =plt.subplots()
    for i in range(n):
        ax.scatter(data[i][0],data[i][1], c=colours[labels[i]], s=20)
    plt.show()
    
    #Split train test
    train_data, test_data, train_labels, test_labels=train_test_split(data, labels, test_size=0.2)
    
    #fit and getmodel
    p=Perceptron(random_state=278612)
    p.fit(train_data,train_labels)
    
    #accuracy score
    predictions_train = p.predict(train_data)
    predictions_test = p.predict(test_data)
    train_score = accuracy_score(predictions_train, train_labels)
    print("score on train data: ", train_score)
    test_score = accuracy_score(predictions_test, test_labels)
    print("score on train data: ", test_score)
    
    #classification report
    print(classification_report(p.predict(train_data), train_labels))
    print(classification_report(p.predict(test_data), test_labels))
    
    #predict point 1,2 -> classified as 0
    print(p.predict([[1,2]]))
    #predict point 5,7 -> classified as 1
    print(p.predict([[5,7]]))