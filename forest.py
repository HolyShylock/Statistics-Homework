from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

def get(tf):
    ones = pd.DataFrame({'ones' : np.ones(len(tf))})
    data = pd.concat([ones, tf], axis = 1)
    X = data.iloc[ : , : -1]
    y = data.iloc[ : , -1]
    return X, y
data = pd.read_csv('heart_failure_clinical_records_dataset1.csv')
X, y = get(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)
scaler = MinMaxScaler()
model = RandomForestClassifier(n_estimators = 1000)
model.fit(X_train, y_train)
y_pre = model.predict(X_test)
accuracy = 'accuracy : ' + str(accuracy_score(y_test, y_pre)) + '\n'
precision = 'precision : ' + str(precision_score(y_test, y_pre)) + '\n'
recall = 'recall : ' + str(recall_score(y_test, y_pre)) + '\n'
print(accuracy + precision + recall)