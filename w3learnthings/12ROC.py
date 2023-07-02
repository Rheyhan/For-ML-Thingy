"""
AUC-ROC CURVE:

ROC curve (receiver operating characteristic curve) is a graph showing the 
performance of a classification model at all classification thresholds.
"""

#Libraries
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

#dummy y
n = 10000
ratio = .95
n_0 = int((1-ratio) * n)
n_1 = int(ratio * n)
y = np.array([0] * n_0 + [1] * n_1)

def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def contoh1():
    y = np.array([0] * n_0 + [1] * n_1)
    # below are the probabilities obtained from a hypothetical model that always predicts the majority class
    # probability of predicting class 1 is going to be 100%
    y_proba = np.array([1]*n)
    y_pred = y_proba > .5

    print("Model 1:")
    print(f'accuracy score: {accuracy_score(y, y_pred)}')
    cf_mat = confusion_matrix(y, y_pred)
    print('Confusion matrix')
    print(cf_mat)
    print(f'class 0 accuracy: {cf_mat[0][0]/n_0}')
    print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')
    plot_roc_curve(y, y_proba)
    print(f'model 1 AUC score: {roc_auc_score(y, y_proba)}\n')

def contoh2():
    y_proba_2 = np.array(
        np.random.uniform(0, .7, n_0).tolist() +
        np.random.uniform(.3, 1, n_1).tolist()
    )
    y_pred_2 = y_proba_2 > .5

    print("Model 2:")
    print(f'accuracy score: {accuracy_score(y, y_pred_2)}')
    cf_mat = confusion_matrix(y, y_pred_2)
    print('Confusion matrix')
    print(cf_mat)
    print(f'class 0 accuracy: {cf_mat[0][0]/n_0}')
    print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')
    plot_roc_curve(y, y_proba_2)
    print(f'model 1 AUC score: {roc_auc_score(y, y_proba_2)}\n')
    
if __name__=="__main__":
    contoh1()
    contoh2()
        
    #Pembahasan
    '''
    Pada model 1, walaupun nilai akurasinya tinggi, namun tidak dapat mengidentifikasi
    class 0. Sehinga model ini cacat

    pada model 2, meskipun nilai akurasi yang didapat lebih rendah daripada
    model 1. Model ini dapat mengidentifikasi class 0

    Sehingga, didapatkan AUC model 2 lebih besar daripada model 1

    '''
    
    '''Prediksi'''
    n = 10000
    y = np.array([0] * n + [1] * n)
    y_prob_1 = np.array(
        np.random.uniform(.25, .5, n//2).tolist() +
        np.random.uniform(.3, .7, n).tolist() +
        np.random.uniform(.5, .75, n//2).tolist()
    )
    y_prob_2 = np.array(
        np.random.uniform(0, .4, n//2).tolist() +
        np.random.uniform(.3, .7, n).tolist() +
        np.random.uniform(.6, 1, n//2).tolist()
    )

    print(f'model 1 accuracy score: {accuracy_score(y, y_prob_1>.5)}')
    print(f'model 2 accuracy score: {accuracy_score(y, y_prob_2>.5)}')

    print(f'model 1 AUC score: {roc_auc_score(y, y_prob_1)}')
    print(f'model 2 AUC score: {roc_auc_score(y, y_prob_2)}')
    plot_roc_curve(y, y_prob_1)
    plt.show()
    fpr, tpr, thresholds = roc_curve(y, y_prob_2)
    plt.plot(fpr, tpr)
    plt.show()
    '''
    Meskipun kedua model prediksi memiliki akurasi yang hampir sama
    tapi lebih bagus jika menggunakan model yang memiliki AUC score
    tinggi terutama ketika memprediksi data selanjutnya
    '''