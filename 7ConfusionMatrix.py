'''
A confusion matrix:
a matrix (table) that can be used to measure the performance of an machine learning
algorithm, usually a supervised learning one. 
Each row of the confusion matrix represents the instances of an
actual class and each column represents the instances of a predicted class. 
But it can be the other way around as well, i.e. rows for predicted classes and columns
for actual classes. 
The name confusion matrix reflects the fact that it makes it easy for us to see what kind of
confusions occur in our classification algorithms. 
For example the algorithms should have predicted a sample as ci because the actual class is ci, but the algorithm came out with cj. In this case of mislabelling the element
cm[i, j] will be incremented by one, when the confusion matrix is constructed.
------------------------------------------------------------------------------------------------------------------------
'''

#Libraries
import numpy as np

class untukconfumatix():
    def __init__(self, confusion_matrix):
        self.confusion_matrix=confusion_matrix
    
    def recall(self, labels):
        row = self.confusion_matrix[labels, :]
        return self.confusion_matrix[labels, labels] / row.sum()

    def precision(self,labels):
        col = self.confusion_matrix[:, labels]
        return self.confusion_matrix[labels, labels] / col.sum()
    
    def precision_macro_average(self):
        rows, columns = self.confusion_matrix.shape
        sum_of_precisions = 0
        for label in range(rows):
            sum_of_precisions += untukconfumatix.precision(self, labels=label)
        return sum_of_precisions / rows

    def recall_macro_average(self):
        rows, columns = self.confusion_matrix.shape
        sum_of_recalls = 0
        for label in range(columns):
            sum_of_recalls += untukconfumatix.recall(self, labels=label)
        return sum_of_recalls / columns

    def accuracy(self):
        diagonal_sum = self.confusion_matrix.trace()
        sum_of_all_elements = self.confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements
    
if __name__ == "__main__":
    dummy = np.array(
    [[5825, 1, 49, 23, 7, 46, 30, 12, 21, 26],
    [ 1, 6654, 48, 25, 10, 32, 19, 62, 111, 10],
    [ 2, 20, 5561, 69, 13, 10, 2, 45, 18, 2],
    [ 6, 26, 99, 5786, 5, 111, 1, 41, 110, 79],
    [ 4, 10, 43, 6, 5533, 32, 11, 53, 34, 79],
    [ 3, 1, 2, 56, 0, 4954, 23, 0, 12, 5],
    [ 31, 4, 42, 22, 45, 103, 5806, 3, 34, 3],
    [ 0, 4, 30, 29, 5, 6, 0, 5817, 2, 28],
    [ 35, 6, 63, 58, 8, 59, 26, 13, 5394, 24],
    [ 16, 16, 21, 57, 216, 68, 0, 219, 115, 5693]])
    
    test=untukconfumatix(dummy)
    print("label precision recall")
    for label in range(10):
        print(f"{label:5d} {test.precision(label):9.3f} {test.recall(label):6.3f}")
        
    print(f'precision_macro_average :   {test.precision_macro_average()}')
    print(f'recall_macro_average    :   {test.recall_macro_average()}')
    print(f'Accuracy                :   {test.accuracy()}')