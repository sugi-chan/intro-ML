
# Load libraries
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cats = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dat = pd.read_csv(url, names=cats)

dat.shape
dat.head()

# descriptions
print(dat.describe())

# class distribution
print(dat.groupby('class').size())


# Split-out validation dataset
array = dat.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7 #for reproducability
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

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
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)




print()
print 'random guess is .4 (12/30)'
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print
print 'K nearest neighbors'
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions,labels=['Iris-setosa','Iris-virginica','Iris-versicolor']))
print(classification_report(Y_validation, predictions))

svc = SVC()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print
print "Support vector machine"
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions,labels=['Iris-setosa','Iris-virginica','Iris-versicolor']))
print(classification_report(Y_validation, predictions))


