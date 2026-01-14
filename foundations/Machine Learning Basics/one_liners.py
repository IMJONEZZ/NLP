#Logistic Regression
model = LogisticRegression().fit(data, labels)

#K-Means Clustering
kmeans = KMeans(n_clusters=2).fit(data)

#K-Nearest Neighbors
KNN = KNeighborsRegressor(n_neighbors=3).fit(data, labels)

#Support Vector Machines
svm = svm.SVC().fit(data, labels)

#Decision Trees
Tree = tree.DecisionTreeClassifier().fit(data, labels)

#Neural Networks
NN = MLPRegressor(max_iter=2000).fit(data, labels)