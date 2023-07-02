"""
Linear Regression:
-Linear regression uses the relationship between the data-points to draw a straight line through all them.
-This line can be used to predict future values.
"""

#contoh
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

df=pd.DataFrame({
    "x": [5,7,8,7,2,17,2,9,4,11,12,9,6],
    "y": [99,86,87,88,111,86,103,87,94,78,77,85,86]
})

def showscatter(df):                                #for the scattrerplot
    plt.scatter(df.iloc[:,0], df.iloc[:,1])

def model(x):                                       #model regresi
    return intercept+ slope * x   #B0 + B1x

def ols(df):                                        #ols method (Linear trend)
    x=df["x"]
    mymodel = list(map(model, x))
    plt.plot(x, mymodel)
    
if __name__ =="__main__":
    showscatter(df)
    slope, intercept, r, p, std_err = stats.linregress(df["x"], df["y"])
    ols(df)
    print(f'corr: {r}\npval: {p}\nmodel: {intercept}{slope}x')
    plt.show()
    
    #predicting x=10
    print(model(10))