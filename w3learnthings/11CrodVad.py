"""
Cross Validation:

Cross-validation is a resampling method that uses different portions of the data to test
and train a model on different iterations. It is mainly used in settings where the goal
is prediction, and one wants to estimate how accurately a predictive model will perform 
in practice
"""

#libraries
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.model_selection import LeavePOut, cross_val_score
from sklearn.model_selection import ShuffleSplit, cross_val_score

X, y = datasets.load_iris(return_X_y=True)
clf = DecisionTreeClassifier(random_state=42)

class method:
    #K-Fold
    '''The training data used in the model is split, into k number of smaller sets, to be
    used to validate the model. The model is then trained on k-1 folds of training set. 
    The remaining fold is then used as a validation set to evaluate the model.
    '''
    def K_Fold():
        k_folds = KFold(n_splits = 5)
        scores = cross_val_score(clf, X, y, cv = k_folds)
        
        print("K-Folds Method:")
        print("Cross Validation Scores: ", scores)
        print("Average CV Score: ", scores.mean())
        print("Number of CV Scores used in Average: ", len(scores),"\n")


    #Stratified K-Fold
    '''stratify the target classes, meaning that both sets will have an equal proportion
    of all classes.'''
    def SKFold():
        sk_folds = StratifiedKFold(n_splits = 5)
        scores = cross_val_score(clf, X, y, cv = sk_folds)

        print("STratified K-Folds Method:")
        print("Cross Validation Scores: ", scores)
        print("Average CV Score: ", scores.mean())
        print("Number of CV Scores used in Average: ", len(scores),"\n")


    #Leave-One-Out (LOO)
    '''Instead of selecting the number of splits in the training data set like k-fold 
    LeaveOneOut, utilize 1 observation to validate and n-1 observations to train. This
    method is an exaustive technique.'''
    def LOO():
        loo = LeaveOneOut()
        scores = cross_val_score(clf, X, y, cv = loo)

        print("Leave-One-Out Method:")
        print("Cross Validation Scores: ", scores)
        print("Average CV Score: ", scores.mean())
        print("Number of CV Scores used in Average: ", len(scores),"\n")

    #Leave-P-Out (LPO)
    '''Leave-P-Out is simply a nuanced diffence to the Leave-One-Out idea, in that we can
    select the number of p to use in our validation set.'''
    def LPO():
        lpo = LeavePOut(p=2)
        scores = cross_val_score(clf, X, y, cv = lpo)

        print("Leave-P-Out Method:")
        print("Cross Validation Scores: ", scores)
        print("Average CV Score: ", scores.mean())
        print("Number of CV Scores used in Average: ", len(scores),"\n") 
    
    #Shuffle Split
    '''ShuffleSplit leaves out a percentage of the data, not to be used in the train or
    validation sets. To do so we must decide what the train and test sizes are, as well
    as the number of splits.'''
    def ShufSplit():
        ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits = 5)
        scores = cross_val_score(clf, X, y, cv = ss)

        print("Shuffle Split Method:")
        print("Cross Validation Scores: ", scores)
        print("Average CV Score: ", scores.mean())
        print("Number of CV Scores used in Average: ", len(scores),"\n")

if __name__  == "__main__":
    method.K_Fold()
    method.SKFold()
    method.LOO()
    method.LPO()
    method.ShufSplit()