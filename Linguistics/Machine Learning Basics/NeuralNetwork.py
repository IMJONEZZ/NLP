from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

df = pd.read_csv('titanic.csv')
df = df.drop(['Name'], axis=1)
df = df.drop(['Ticket'], axis=1)
df = df.drop(['Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(method ='pad')
d = {'male':0, 'female':1}
df['Sex']=df['Sex'].apply(lambda x:d[x])
e={'C':0, 'Q':1 ,'S':2}
df['Embarked']=df['Embarked'].apply(lambda x:e[x])

test_df = pd.read_csv('test.csv')
test_df = test_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)
test_df['Age'] = test_df['Age'].fillna(df['Age'].median())
test_df['Embarked'] = test_df['Embarked'].fillna(method ='pad')
test_df['Sex']=test_df['Sex'].apply(lambda x:d[x])
test_df['Embarked']=test_df['Embarked'].apply(lambda x:e[x])

X = df.to_numpy()
test_X = test_df.to_numpy()

NN = MLPRegressor(max_iter=2000).fit(X[:,:-1], X[:,-1]) 

results = []
for entry in X:
    res = NN.predict(entry[:-1].reshape(1, -1))
    if res[0] > 0.5:
        results.append(1)
    if res[0] < 0.5:
        results.append(0)
print("Accuracy Score on training set: ",accuracy_score(results, X[:,-1]))
print("Confusion Matrix:\n",confusion_matrix(results, X[:,-1]))

results = []
for entry in test_X:
    res = NN.predict(entry.reshape(1, -1))
    if res[0] > 0.5:
        results.append(1)
    if res[0] < 0.5:
        results.append(0)
solution = pd.read_csv('solution.csv')
print("Accuracy Score on real test data: ",accuracy_score(results, solution['survived']))
print("Confusion Matrix:\n",confusion_matrix(results, solution['survived']))