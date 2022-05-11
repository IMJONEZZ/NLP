from sklearn.cluster import KMeans
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

X = df.to_numpy()

model = KMeans(5).fit(X[:,:-1], X[:,-1])

cc = model.cluster_centers_
print(cc)