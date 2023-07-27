#Multilayer logistical regression example
from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y = [0, 0, 0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), #2 hidden layers
                    random_state=1)
print(clf.fit(X, y))
'''
The attribute coefs_ contains a list of
weight matrices for every layer. The
weight matrix at index i holds the weights
between the layer i and layer i + 1.
'''
print("weights between input and first hidden layer:")
print(clf.coefs_[0])
print("\nweights betweenfirst hidden and second hidden layer:")
print(clf.coefs_[1])

#get weight values
print("w0 = ", clf.coefs_[0][0][0])
print("w1 = ", clf.coefs_[0][1][0])

#access a neuron Hij
for i in range(len(clf.coefs_)):
    number_neurons_in_layer = clf.coefs_[i].shape[1]
    for j in range(number_neurons_in_layer):
        weights = clf.coefs_[i][:,j]
        print(i, j, weights, end=", \n")
    print("")

#intercepts_ is a list of bias vectors, where the vector at index i represents the bias values added to layer i+1.
print("Bias values for first hidden layer:")
print(clf.intercepts_[0])
print("\nBias values for second hidden layer:")
print(clf.intercepts_[1])

#predict
result = clf.predict([[0, 0], [0, 1],
                      [1, 0], [0, 1],
                      [1, 1], [2., 2.],
                      [1.3, 1.3], [2, 4.8]])
print(result)

#probability estimates
prob_result = clf.predict_proba([[0, 0], [0, 1],
                      [1, 0], [0, 1],
                      [1, 1], [2., 2.],
                      [1.3, 1.3], [2, 4.8]])
print(prob_result)