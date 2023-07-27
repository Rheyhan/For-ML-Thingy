import pickle
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
if __name__ == "__main__":
    with open("Files\pickled_mnist.pkl", "br") as fh:
        data=pickle.load(fh)
    train_imgs = data[0]
    test_imgs = data[1]
    train_labels = data[2]
    test_labels = data[3]
    train_labels_one_hot = data[4]
    test_labels_one_hot = data[5]
    
    no_of_different_labels = 10
    image_pixels = 28**2

    ANN = MLPClassifier(hidden_layer_sizes=(100, ),
                        max_iter=480, alpha=1e-4,
                        solver='sgd', verbose=10,
                        tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    ANN.fit(train_imgs, train_labels)
    
    print("Training set score: %f" % ANN.score(train_imgs, train_labels))
    print("Test set score: %f" % ANN.score(test_imgs, test_labels))
    
    fig, axes = plt.subplots(4, 4)
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = ANN.coefs_[0].min(), ANN.coefs_[0].max()
    for coef, ax in zip(ANN.coefs_[0].T, axes.ravel()):
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())
    plt.show()