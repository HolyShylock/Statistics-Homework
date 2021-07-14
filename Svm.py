from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
import numpy as np

def get(tf):
    ones = pd.DataFrame({'ones' : np.ones(len(tf))})
    data = pd.concat([ones, tf], axis = 1)
    X = data.iloc[ : , : -1]
    y = data.iloc[ : , -1]
    return X, y
def rate(y_pre, y_test):
    y_test = np.matrix(y_test)
    y_pre = np.matrix(y_pre)
    count = 0.000
    n = y_test.shape[1]
    for i in range(0, n):
        if y_pre[0, i] == y_test[0, i]:
            count += 1
    res = 'accuracy rate: ' + str(count / n)
    print(res)

data = pd.read_csv('heart_failure_clinical_records_dataset1.csv')
X, y = get(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
model = svm.SVC(kernel = 'linear', C = 1, gamma = 2)
model.fit(X_train, y_train)
pre = model.predict(X_test)
rate(pre, y_test)