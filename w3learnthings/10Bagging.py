'''
Bootstrap Aggregation (bagging):

an ensembling method that attempts to resolve overfitting for classification or 
regression problems. Bagging aims to improve the accuracy and performance of machine 
learning algorithms. It does this by taking random subsets of an original dataset, with 
replacement, and fits either a classifier (for classification) or regressor (for 
regression) to each subset. The predictions for each subset are then aggregated through 
majority vote for classification or averaging for regression, increasing prediction 
accuracy.
'''
#Intinya meningkatkan akurasi klasifikasi atau regresi tree


#libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#dataframe
df = datasets.load_wine(as_frame = True)

def inipakeDecisionTreeClassifier(df):
    X=df.data
    y=df.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)

    dtree = DecisionTreeClassifier(random_state = 22) 
    dtree.fit(X_train,y_train)

    y_pred = dtree.predict(X_test)

    print("Train data accuracy:",accuracy_score(y_true = y_train, y_pred = dtree.predict(X_train)))
    print("Test data accuracy:",accuracy_score(y_true = y_test, y_pred = y_pred))


def inipakeBaggingClassifier(df):
    X=df.data
    y=df.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)
    estimator_range = [2,4,6,8,10,12,14,16]
    models = []
    scores = []
    for n_estimators in estimator_range:
        # Create bagging classifier
        clf = BaggingClassifier(n_estimators = n_estimators, random_state = 22)
        # Fit the model
        clf.fit(X_train, y_train)
        # Append the model and score to their respective list
        models.append(clf)
        scores.append(accuracy_score(y_true = y_test, y_pred = clf.predict(X_test)))

    # Generate the plot of scores against number of estimators
    plt.figure(figsize=(9,6))
    plt.plot(estimator_range, scores)
    # Adjust labels and font (to make visable)
    plt.xlabel("n_estimators", fontsize = 18)
    plt.ylabel("score", fontsize = 18)
    plt.tick_params(labelsize = 16)
    # Visualize plot
    plt.show()

def visualizeBaggingClassifier(df):
    X=df.data
    y=df.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)
    clf = BaggingClassifier(n_estimators = 12, oob_score = True,random_state = 22)  #n_estimator=12
    clf.fit(X_train, y_train)
    plt.figure(figsize=(10, 10))
    plot_tree(clf.estimators_[0], feature_names = X.columns)
    plt.show()

if __name__ == "__main__":
    inipakeDecisionTreeClassifier(df)
    inipakeBaggingClassifier(df)
    visualizeBaggingClassifier(df)