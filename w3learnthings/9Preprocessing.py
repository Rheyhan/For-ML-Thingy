"""
Praprocessing:
Intinya, menggunakan data jenis kategorik pada regresi agar memperjelas suatu prediksi
"""

#libraries
import pandas as pd
import numpy as np
from sklearn import linear_model

df=pd.DataFrame({
    "Car"   :["Toyota","Mitsubishi","Skoda","Fiat","Mini","VW","Skoda","Mercedes","Ford","Audi","Hyundai","Suzuki","Ford","Honda","Hundai","Opel","BMW","Mazda","Skoda","Ford","Ford","Opel","Mercedes","Skoda","Volvo","Mercedes","Audi","Audi","Volvo","BMW","Mercedes","Volvo","Ford","BMW","Opel","Mercedes"],
    "Model" :["Aygo","Space Star","Citigo","500","Cooper","Up!","Fabia","A-Class","Fiesta","A1","I20","Swift","Fiesta","Civic","I30","Astra","1","3","Rapid","Focus","Mondeo","Insignia","C-Class","Octavia","S60","CLA","A4","A6","V70","5","E-Class","XC70","B-Max","2","Zafira","SLK"],
    "Volume":[1000,1200,1000,900,1500,1000,1400,1500,1500,1600,1100,1300,1000,1600,1600,1600,1600,2200,1600,2000,1600,2000,2100,1600,2000,1500,2000,2000,1600,2000,2100,2000,1600,1600,1600,2500],
    "Weight":[790,1160,929,865,1140,929,1109,1365,1112,1150,980,990,1112,1252,1326,1330,1365,1280,1119,1328,1584,1428,1365,1415,1415,1465,1490,1725,1523,1705,1605,1746,1235,1390,1405,1395],
    "CO2"   :[99,95,95,90,105,105,90,92,98,99,99,101,99,94,97,97,99,104,104,105,94,99,99,99,99,102,104,114,109,114,115,117,104,108,109,120]
})


if __name__=="__main__":
    '''One Hot Encoding'''
    ohe_df=pd.get_dummies(df.Car)
    #Column representing each group in the category.
    #For each column, the values will be 1 or 0 where 1 represents the inclusion of
    # the group and 0 represents the exclusion.
    
    X=(pd.concat([df[["Volume","Weight"]], ohe_df], axis=1)).values  #mirip kek cbind
    y=df.CO2
    
    model=linear_model.LinearRegression()
    model.fit(X,y)
    
    '''Predicting'''
    #Misal prediksi emisi C02 Mobil Toyota berat 3200kg dan volumenya 1500cm
    predicted=model.predict([[3200,1500,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    print(f'C02 yang dihasilkan :   {predicted}')
    
#'Cara Lain!
    '''Dummifying'''
    colors = pd.DataFrame({'color': ['blue', 'red', 'green']})
    dummies = pd.get_dummies(colors, drop_first=True)
    dummies['color'] = colors['color']

    print(dummies)