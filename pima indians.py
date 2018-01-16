# Rescale data (between 0 and 1)
import pandas as pd
import scipy
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
dataframe.shape
dataframe.head()
array = dataframe.values


# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]

validation_size = 0.20
seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


#rescale between 0 and 1. when you have different scales, it means models may make incorrect assumptions about data
# for example, Kmeans clustering essentially calculates euclidean distance between two points,
# but something like wage and age would be scaled very differently, but the model doesnt know that... so unless
# you scale it the variables will not be weighed correctly

#if you normalize before, you incoperate the validation values into the training set (bleeding)
scalerx = StandardScaler().fit(X_train)
rescaledX = scalerx.transform(X_train)

scaler_validation = StandardScaler()
house_cats = scaler_validation.fit(X_validation)
rescaledX_val = house_cats.transform(X_validation)

# summarize transformed data
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('GBC',GradientBoostingClassifier()))
# evaluate each model in turn

for name, item in [('raw training'),X_train),('rescaled' ,rescaledX]:

    print 'currently working with '+name

    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, item, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)



'''
currently working with raw training data
LR: 0.776864 (0.060738)
KNN: 0.710153 (0.064599)
CART: 0.677552 (0.044677)
SVM: 0.656293 (0.044581)
GBC: 0.744157 (0.084169)
currently working with rescale data (mean of 0 std 1)
LR: 0.780037 (0.061129)
KNN: 0.713432 (0.067707)
CART: 0.679138 (0.057200)
SVM: 0.750925 (0.063180)
GBC: 0.747435 (0.082235)


'''


'''
          Confusion Matrix
            Pred    Pred
            No      Yes
            _____________
Actual NO  |   T  |  F   |
           |______|______|
Actual Yes |   F  |  T   |
           |      |      |
           ---------------
'''

print 'random guess is .63 (97/154)'
# Make predictions on validation dataset
lr = LogisticRegression()
lr.fit(rescaledX, Y_train)
predictions = lr.predict(rescaledX_val)
print
print 'K nearest neighbors'
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

svc = SVC()
svc.fit(rescaledX, Y_train)
predictions = svc.predict(rescaledX_val)
print
print "Support vector machine"
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
