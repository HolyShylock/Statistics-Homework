import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def get_X(df):
    return np.array(df.iloc[ : , : -2])
def get_y(df):
    return np.array([df.iloc[ : , -1]]).reshape(299, 1)
def get(tf):
    ones = pd.DataFrame({'ones' : np.ones(len(tf))})
    data = pd.concat([ones, tf], axis = 1)
    X = data.iloc[ : , : -1]
    y = data.iloc[ : , -1]
    return X, y
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
'''X = get_X(data)
y = get_y(data)
print(X.shape)
print(y.shape)'''
X, y = get(data)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=4)
print(X_train.shape)
print(y_train.shape)
clf = LogisticRegression()
print(clf.fit(X_train, y_train))
y_pre = clf.predict(X_test)
accuracy = 'accuracy : ' + str(accuracy_score(y_test, y_pre)) + '\n'
precision = 'precision : ' + str(precision_score(y_test, y_pre)) + '\n'
recall = 'recall : ' + str(recall_score(y_test, y_pre)) + '\n'
print(accuracy + precision + recall)