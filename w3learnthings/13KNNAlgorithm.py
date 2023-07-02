'''
K-nearest neighbors (KNN):
supervised machine learning (ML) algorithm that can be used for classification or 
regression tasks - and is also frequently used in missing value imputation. It is based
on the idea that the observations closest to a given data point are the most "similar" 
observations in a data set, and we can therefore classify unforeseen points based on the
values of the closest existing points.
'''

'''
K is the number of nearest neighbors to use
Larger values of K are often more robust to outliers and produce more stable decision 
boundaries than very small values (K=3 would be better than K=1, which might produce 
undesirable results.
'''
#--------------------------------------------------------------------------------------
#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#Get data
df=pd.DataFrame({
"x":[4, 5, 10, 4, 3, 11, 14 , 8, 10, 12],
"y":[21, 19, 24, 17, 16, 25, 24, 22, 21, 21],
"classes": [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]
})
x=((df.x).values).tolist()
y=((df.y).values).tolist()
data = list(zip(x, y))
classes=((df.classes).values).tolist()

#Fit Model of KNN
knn = KNeighborsClassifier(n_neighbors=1)   #k=1
knn.fit(data, classes)


def forpredict(xpred,ypred):
    new_point = [(xpred, ypred)]

    #predict the new coordinate
    prediction = knn.predict(new_point)

    plt.scatter((x+[xpred]),( y + [ypred]), c=(classes + [prediction[0]]))
    plt.text(x=xpred-1.7, y=ypred-0.7, s=f"new point, class: {prediction[0]}")
    plt.show()


if __name__ =="__main__":
    #plot scatter cluster
    plt.scatter(x,y,c=classes)
    plt.show()
    
    forpredict(8,21)            #predicting what (8,21) class and will also get plotted    
    

#----------------------------------------------------------------------------------------

#coba-coba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X=[0,1,2,3]
y=[0,0,1,1]
classes=[0,0,1,1]

data=list(zip(X,y))

plt.scatter(X,y,c=classes)
plt.show()

model=KNeighborsClassifier(n_neighbors=1)
model.fit(data,classes)


prediction=[(3,2)]
print(prediction)
this=model.predict(prediction)
print(this[0])
plt.scatter(X+[3], y+[2], c=(classes+[this[0]]))
plt.show()
