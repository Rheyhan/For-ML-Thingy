from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

if __name__ == "__main__":
    #getting the data
    iris = load_iris()
    data=iris.data
    labels=iris.target
    #Preparing train test split
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels,
                                                                        random_state=23712,
                                                                        test_size=0.2)
        #scale
    scale=StandardScaler()
    scale.fit(train_data)                      #fit the scale
    train_data=scale.transform(train_data)
    test_data=scale.transform(test_data)
    #Create and train neural network model
    ANN=MLPClassifier(hidden_layer_sizes=(10,5), max_iter=1000)
    ANN.fit(train_data, train_labels)
    
    #getting score Automatically
    print(f'Train:  {ANN.score(train_data, train_labels)}')
    print(f'Test :  {ANN.score(test_data, test_labels)}')
    
    #Getting score Manually
    TrainPredicted=ANN.predict(train_data)
    TestPredicted=ANN.predict(test_data)
    print(f'Train:  {accuracy_score(TrainPredicted, train_labels)}')
    print(f'Test:   {accuracy_score(TestPredicted, test_labels)}')
    
    #confusion matrix
    print("Train:")
    print(confusion_matrix(TrainPredicted, train_labels))
    print("Test:")
    print(confusion_matrix(TestPredicted, test_labels))    
    
    #classification report
    print(classification_report(TestPredicted, test_labels))
          
    #predicting things/ Run for fun
    for i in range(20):
        z=ANN.predict((test_data[i].reshape(1,-1)))
        print(f'Actual:{test_labels[i]}|  Prediction: {z} ')