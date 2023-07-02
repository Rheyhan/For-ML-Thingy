"""
Logistic Regression:

Logistic regression aims to solve classification problems. It does this by predicting categorical outcomes, unlike linear regression that predicts a continuous outcome.
"""

import pandas as pd
import numpy as np
from sklearn import linear_model

df=pd.DataFrame({
    "x": [3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88],
    "y": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    })
#x: panjang tumor (cm) && y: Apakah tumor itu berkanker (0:Tidak, 1: Iya)


def logit2prob(logr,x):
  log_odds = logr.coef_ * x + logr.intercept_ #b0x+b1
  odds = np.exp(log_odds)                     #Ubah exponen (karena dari log)
  probability = odds / (1 + odds)
  return(probability)


if __name__ == "__main__":
    x=np.array(df.x).reshape(-1,1)                          #change to 2d array
    model=linear_model.LogisticRegression()
    model.fit(x,df.y)                                       #getmodel
    
    print(f'Rsq     :   {model.score(x,df.y)}')             #R square
    
    '''coefficient'''
    coefficient = np.exp(model.coef_)  #In logistic regression the coefficient is the expected change in log
    print(coefficient)
    #This tells us that as the size of a tumor increases by 1mm the odds of it being a tumor increases by 4x.
    
    '''Predicting   (Predict x=6)''' 
    print(f'y6      :{model.predict(np.array(6).reshape(-1,1))}')#Ya ini tumor kanker
    
    '''Probability'''
    print(logit2prob(model,x))
    #Maksud:
    #3.78 0.61 The probability that a tumor with the size 3.78cm is cancerous is 61%.
    #2.44 0.19 The probability that a tumor with the size 2.44cm is cancerous is 19%.
    #2.09 0.13 The probability that a tumor with the size 2.09cm is cancerous is 13%.