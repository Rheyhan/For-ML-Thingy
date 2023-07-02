'''
Multiple Logistic Regression:
a statistical test used to predict a single binary variable using one or more other variables. It also is used to determine the numerical relationship between such a set of variables.
'''

#libraries
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

df=datasets.load_iris()
X=df.data     #Sepal length and width, petal length and width
y=df.target   #Species

if __name__ == '__main__':
  model= LogisticRegression(max_iter=6940)  #c default=1
  #C: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization

  model.fit(X,y)
  print(f'Rsq   : {model.score(X,y)}')                              #Rsq
  
  '''Grid Search'''
  C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
  scores = []
  for choice in C:
    model.set_params(C=choice)
    model.fit(X, y)
    scores.append(model.score(X, y))

  print(scores) #akurasi bagus pada 1.75