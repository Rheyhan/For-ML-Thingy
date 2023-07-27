from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

n=200
blob_centers = ([1, 1], [3, 4], [1, 3.3], [3.5, 1.8])
data, labels = make_blobs(n_samples=n,
                          centers=blob_centers,
                          cluster_std=0.5,
                          random_state=0)
colours = ('green', 'orange', "blue", "magenta")
fig, ax = plt.subplots()

for i in range(len(blob_centers)):
    plt.scatter(data[labels==i][:,0], data[labels==i][:,1], s=20, c=colours[i], label=str(i))
plt.show()

train_data, test_data, train_labels, test_labels= train_test_split(data, labels, test_size=0.2)

from sklearn.neural_network import MLPClassifier
'''
• hidden_layer_sizes: tuple, length = n_layers - 2, default=(100,)
The ith element represents the number of neurons in the ith hidden layer.
(6,) means one hidden layer with 6 neurons

• solver:
The weight optimization can be influenced with the solver parameter. Three solver modes
are available
    ▪ 'lbfgs'
    is an optimizer in the family of quasi-Newton methods.
    ▪ 'sgd'
    refers to stochastic gradient descent.
    ▪ 'adam' refers to a stochastic gradient-based optimizer proposed by Kingma,
    Diederik, and Jimmy Ba
Without understanding in the details of the solvers, you should know the following: 'adam'
works pretty well - both training time and validation score - on relatively large datasets, i.e.
thousands of training samples or more. For small datasets, however, 'lbfgs' can converge faster
and perform better.

• 'alpha'
This parameter can be used to control possible 'overfitting' and 'underfitting'.
'''
clf = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(6,),
                    random_state=1)
clf.fit(train_data, train_labels)

print(clf.score(train_data, train_labels))
print(clf.score(test_data, test_labels))

#manual score
from sklearn.metrics import accuracy_score
predictions_train = clf.predict(train_data)
predictions_test = clf.predict(test_data)
train_score = accuracy_score(predictions_train, train_labels)
print("score on train data: ", train_score)
test_score = accuracy_score(predictions_test, test_labels)
print("score on train data: ", test_score)