"""
Train.Test:

In Machine Learning we create models to predict the outcome of certain events, like in the previous chapter where we predicted the CO2 emission of a car when we knew the weight and engine size.

To measure if the model is good enough, we can use a method called Train/Test.
"""


"""
Train the model means create the model.
Test the model means test the accuracy of the model.

You train the model using the training set.
You test the model using the testing set.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
np.random.seed(69720)

class inidata:
    x= np.random.normal(3,1,100)
    y= np.random.normal(150,40,100)/x
    df=pd.DataFrame({
        "x": x,
        "y": y
    })

def getmodel(X_train,y_train):                          #polynomial regression
    model = np.poly1d(np.polyfit(X_train, y_train, 4))
    myline = np.linspace(0, 6, 100)
    #scayer and its polynomial trendline
    plt.scatter(X_train, y_train)
    plt.plot(myline, model(myline))
    return model

if __name__ == "__main__":
    df=inidata.df
    X_train, X_test, y_train, y_test = train_test_split(df.x, df.y,train_size=0.8)  #splitting train and test, train=0.8
    model=getmodel(X_train,y_train)
    plt.show()
    
    #show model
    print(model)
    #find rsq
    rsq=r2_score(y_train, model(X_train))
    print(f'rsq :   {rsq}')
    
    #predicting value, x=3.2
    predict=model(3.2)
    print(f'y3.2 :   {predict}')