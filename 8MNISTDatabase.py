'''
The MNIST database:
(Modified National Institute of Standards and Technology database) of handwritten
digits consists of a training set of 60,000 examples,
and a test set of 10,000 examples. It is a subset of a
larger set available from NIST. Additionally, the
black and white images from NIST were sizenormalized
and centered to fit into a 28x28 pixel
bounding box and anti-aliased, which introduced
grayscale levels.
'''
#Libraries
import numpy as np
import matplotlib.pyplot as plt

img_size=28**2
diff_labels=10      #0-9 (10)

#load data
train_data=np.loadtxt("mnist_train.csv", delimiter=",", skiprows=1 )
test_data=np.loadtxt("mnist_test.csv", delimiter=",", skiprows=1 )

#print(test_data[:10])                                                  #check if applied correctly
#print(test_data.shape)

'''
The images of the MNIST dataset are greyscale and the pixels range between 0 and 255 including both
bounding values. We will map these values into an interval from [0.01, 1] by multiplying each pixel by 0.99 /
255 and adding 0.01 to the result.
'''

#check one hot reprentation
lr = np.arange(10)
for label in range(10):
    one_hot = (lr==label).astype(int)
    print("label: ", label, " in one-hot representation: ", one_hot)


fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

lr = np.arange(diff_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(float)
test_labels_one_hot = (lr==test_labels).astype(float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99


def showtenimages():
    for i in range(10):
        img = train_imgs[i].reshape((28,28))
        plt.imshow(img, cmap="Greys")
        plt.show()
        
#showtenimages()
    
    
#continue to pickle method! (cuz this sh is slow af)
import pickle

with open("Files/pickled_mnist.pkl", "bw") as fh:
    data = (train_imgs,
            test_imgs,
            train_labels,
            test_labels,
            train_labels_one_hot,
            test_labels_one_hot)
    pickle.dump(data, fh)
