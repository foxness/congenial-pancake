import numpy as np
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FILENAME = "wifi_localization.txt"

# reading the dataset

lines = [line.rstrip('\n').split('\t') for line in open(FILENAME)]
lines = [[int(x) for x in line] for line in lines]

dataset = {}
dataset['data'] = np.array([line[:-1] for line in lines])
dataset['target'] = np.array([line[-1] for line in lines])

# splitting the dataset into training and testing parts

X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state=0)

# training the logistic regression model

logreg = LogisticRegression(C=10, solver='lbfgs', max_iter=10000, multi_class='auto')
logreg.fit(X_train, y_train)

# showing the model score

print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))