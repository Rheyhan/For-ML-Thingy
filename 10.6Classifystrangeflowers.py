import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

if __name__ == "__main__":
    df=pd.read_table("Files/Flowerthingy.txt", sep=" ", header=None)
    Data=df.iloc[:,0:3].values
    labels=df.iloc[:,4].values
    
    Data_Train, Data_Test, Labels_Train, Labels_Test = train_test_split(Data, labels, train_size=0.8, random_state=2376478)
    
    #we go scale
    scale=StandardScaler()
    Data_Train=scale.fit_transform(Data_Train)
    Data_Test=scale.transform(Data_Test)
    
    #Get model
    ANN = MLPClassifier(hidden_layer_sizes=(100, 100),
                        max_iter=480, alpha=1e-4,
                        solver='sgd', verbose=10,
                        tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    
    #train
    ANN.fit(Data_Train, Labels_Train)
    
    #get score of both train and test
    print(f'Train Score:    {ANN.score(Data_Train, Labels_Train):.5f}')
    print(f'Test Score :    {ANN.score(Data_Test, Labels_Test):.5f}')
    
    
    #Get predicted
    trainpredicted=ANN.predict(Data_Train)
    testpredicted=ANN.predict(Data_Test)
    
    #Confusion Matrix
    print("Train:   ")
    print(confusion_matrix(Labels_Train, trainpredicted))
    print("Test:    ")
    print(confusion_matrix(Labels_Test, testpredicted))
    
    #classification_report
    print(classification_report(Labels_Test, testpredicted))