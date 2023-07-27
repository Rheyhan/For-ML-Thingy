from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def show64images(data, labels):
    fig=plt.figure(figsize=(6,6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    j=1
    for i in range(64):
        ax = fig.add_subplot(8, 8, j)
        ax.imshow(data[i], cmap="binary", interpolation='nearest')
        j+=1
        ax.text(0,7, str(labels[i]))
    plt.show()

if __name__ == "__main__":
    digits=load_digits()
    colnames=list(digits.keys())
    print(digits.images[0])
    print(digits.data[0])
    data=digits.data
    n_samples, n_features = digits.data.shape       #n=1797 |   Variables:  64
    #print(digits.target)                           #0-9
    
    #Making sure of the shape
    print("Length and wide  : ", digits.images[0].shape)
    print("Image res size   : ", digits.data[0].shape)
    
    #testing images and its labels
    #show64images(digits.images, digits.target)
    
    #split train test
    train_data, test_data, train_labels, test_labels=train_test_split(data, digits.target, train_size=0.7, random_state=112122)
    
    #Get model
    ANN=MLPClassifier(activation="logistic", hidden_layer_sizes=(100,100),
                      solver="sgd", learning_rate_init=.3, random_state=3726327,
                      tol=1e-4, alpha=1e-4, verbose=True)
    #fit
    ANN.fit(train_data, train_labels)
    
    #score
    print(f'Train:  {ANN.score(train_data, train_labels):.5f}')
    print(f'Test:   {ANN.score(test_data, test_labels):5f}')
    
    #predicted
    testpredicted=ANN.predict(test_data)    
    #confusion Matrix
    print(confusion_matrix(test_labels, testpredicted))
    #classification report
    print(classification_report(test_labels, testpredicted))
    
    #show random images to show how accurate it is
    fig=plt.figure(figsize=(8,8))
    fig.subplots_adjust(0, 0, 1, 1, 0.05, 0.05)
    for i in range(100):
        ax=fig.add_subplot(10,10, i+1)
        yes=test_data[i].reshape(8,8)
        ax.imshow(yes, cmap="binary", interpolation="bilinear")
        ax.text(0,5, str(test_labels[i]),bbox=dict(facecolor='green'))
        if (test_labels[i]!=testpredicted[i]):
            ax.text(0,7, str(testpredicted[i]), bbox=dict(facecolor="red"))
        else:
            ax.text(0,7, str(testpredicted[i]), bbox=dict(facecolor='yellow'))
    plt.show()