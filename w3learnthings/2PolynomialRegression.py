"""
Polynomial Regression:
If your data points clearly will not fit a linear regression (a straight line through all data points), it might be ideal for polynomial regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd


df=pd.DataFrame({
    "x":[1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22],
    "y": [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
})

def showscatter(df):                                #scatterplot
    plt.scatter(df["x"],df["y"])   

def polynomialline(df):                             #polynomial trend
    model= np.poly1d(np.polyfit(df["x"],df["y"],3))#getting model
    theline=np.linspace(min(df["x"]),max(df["x"]),100)
    plt.plot(theline, model(theline))
    return model
    
if __name__=="__main__":
    showscatter(df)
    model=polynomialline(df)
    print(f'model: {model}')
    plt.show()
    
    #predicting value, ex: x=27
    estimation=model(27)
    print(f'y(27): {estimation}')
    
    #finding rsq
    rsq=r2_score(df["y"], model(df["x"]))
    print(f'r^2: {rsq}')