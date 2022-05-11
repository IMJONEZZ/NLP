from sklearn import tree
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

Tree = tree.DecisionTreeClassifier().fit(X[:,:-1], X[:,-1])



res = Tree.predict(X[:,:-1])
print("Accuracy Score on training set: ",accuracy_score(res, X[:,-1]))
print("Confusion Matrix:\n",confusion_matrix(res, X[:,-1]))

res = Tree.predict(test_X)
solution = pd.read_csv('solution.csv')
print("Accuracy Score on real test data: ",accuracy_score(res, solution['survived']))
print("Confusion Matrix:\n",confusion_matrix(res, solution['survived']))