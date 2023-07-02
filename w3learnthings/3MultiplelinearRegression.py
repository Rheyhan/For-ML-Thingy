"""
Multiple Linear Regression:
Multiple regression is like linear regression, but with more than one independent value, meaning that we try to predict a value based on two or more variables.
"""

import pandas as pd
from sklearn import linear_model

df=pd.DataFrame({
    "Car"   :["Toyota","Mitsubishi","Skoda","Fiat","Mini","VW","Skoda","Mercedes","Ford","Audi","Hyundai","Suzuki","Ford","Honda","Hundai","Opel","BMW","Mazda","Skoda","Ford","Ford","Opel","Mercedes","Skoda","Volvo","Mercedes","Audi","Audi","Volvo","BMW","Mercedes","Volvo","Ford","BMW","Opel","Mercedes"],
    "Model" :["Aygo","Space Star","Citigo","500","Cooper","Up!","Fabia","A-Class","Fiesta","A1","I20","Swift","Fiesta","Civic","I30","Astra","1","3","Rapid","Focus","Mondeo","Insignia","C-Class","Octavia","S60","CLA","A4","A6","V70","5","E-Class","XC70","B-Max","2","Zafira","SLK"],
    "Volume":[1000,1200,1000,900,1500,1000,1400,1500,1500,1600,1100,1300,1000,1600,1600,1600,1600,2200,1600,2000,1600,2000,2100,1600,2000,1500,2000,2000,1600,2000,2100,2000,1600,1600,1600,2500],
    "Weight":[790,1160,929,865,1140,929,1109,1365,1112,1150,980,990,1112,1252,1326,1330,1365,1280,1119,1328,1584,1428,1365,1415,1415,1465,1490,1725,1523,1705,1605,1746,1235,1390,1405,1395],
    "CO2"   :[99,95,95,90,105,105,90,92,98,99,99,101,99,94,97,97,99,104,104,105,94,99,99,99,99,102,104,114,109,114,115,117,104,108,109,120]
})

def getmodel(X,y):                              #Get multiple linear model
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    print(f'Rsq         :   {regr.score(X,y)}') #Show Rsq
    return regr

if __name__ == "__main__":
    X=df[["Weight","Volume"]].values   #misal ambil x1 weight dan x2 volume
    y=df.CO2.values
    model=getmodel(X,y)
    print(f'Intercept   :   {model.intercept_}')
    print(f'coefficient :   {model.coef_}')
    
    #predicting, misal x1=2300 dan x2=1300
    print(model.predict([[2300, 1300]]))