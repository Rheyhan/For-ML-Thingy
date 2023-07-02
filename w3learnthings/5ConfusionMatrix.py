"""
confusion matrix:
table that is used in classification problems to assess where errors in the model were made.

The rows represent the actual classes the outcomes should have been. While the columns represent 
the predictions we have made. Using this table it is easy to see which predictions are wrong.

Confusion matrixes can be created by predictions made from a logistic regression.
"""
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

df=pd.DataFrame({
    "actual":np.random.binomial(1, 0.9, size = 1000),
    "predicted" : np.random.binomial(1, 0.9, size = 1000)
})

def theplot(df):      #show confusion matrix plot
    confusion_matrix = metrics.confusion_matrix(df.actual, df.predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels = [False, True])
    cm_display.plot()

def measuremetrics(df):
    #measuring metrics
    #1. accuracy ((True Positive + True Negative) / Total Predictions)
    Accuracy = metrics.accuracy_score(df.actual, df.predicted)
    print(f'accuracy method     :   {Accuracy}')
    
    #2. Precision (True Positive / (True Positive + False Positive))
    Precision = metrics.precision_score(df.actual, df.predicted)
    print(f'precision method    :   {Precision}')
    
    #3 Sensitivity  (True Positive / (True Positive + False Negative))
    Sensitivity_recall = metrics.recall_score(df.actual, df.predicted)
    print(f'sensitivity method  :   {Sensitivity_recall}')
    
    #4. Specifity (True Negative / (True Negative + False Positive))
    Specificity = metrics.recall_score(df.actual, df.predicted, pos_label=0)
    print(f'Specificity method  :   {Specificity}')
    
    #5. f-Score    (2 * ((Precision * Sensitivity) / (Precision + Sensitivity)))
    F1_score = metrics.f1_score(df.actual, df.predicted)
    print(f'F1_score method     :   {F1_score}')
    
if __name__ == "__main__":
    theplot(df)
    measuremetrics(df)
    plt.show()  