import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import bz2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# read in csv data
df = pd.read_csv("datasets/winequality-white.csv", delimiter=";")

# targets all columns except for the last one (quality)
for label in df.columns[:-1]:
    plt.boxplot([df[df["quality"] == i][label] for i in range(0, 11)])
    plt.title(label)
    plt.xlabel("Quality")
    plt.ylabel(label)
    # save image of figure to imgs folder
    plt.savefig("figs/" + "white".join(label.split(" ")))

# split data into training and testing sets
bins = [0, 5.5, 7.5, 10]
# 0 = bad, 1 = average, 2 = good
labels = [0, 1, 2]
df["quality"] = pd.cut(df["quality"], bins = bins, labels = labels)
print(df.head(5))

# scale data to fit between 0 and 2
x = df[df.columns[:-1]]
y = df["quality"]
# removes mean and scales to unit variance
sc = StandardScaler()
x = sc.fit_transform(x)

# training set contains 80% of data, testing set contains 20% of data, random_state = 42 ensures that the same data is used for each run
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# knn classifier with 3 neighbors
n3 = KNeighborsClassifier(n_neighbors = 3)
n3.fit(x_train, y_train)
pred_n3 = n3.predict(x_test)
print(classification_report(y_test, pred_n3))
cross_val = cross_val_score(estimator = n3, X = x_train, y = y_train, cv = 10)
print("KNN3 Accuracy = " + str(cross_val.mean() * 100) + "%")

# knn classifier with 5 neighbors
n5 = KNeighborsClassifier(n_neighbors = 5)
n5.fit(x_train, y_train)
pred_n5 = n5.predict(x_test)
print(classification_report(y_test, pred_n5))
cross_val = cross_val_score(estimator = n5, X = x_train, y = y_train, cv = 10)
print("KNN5 Accuracy = " + str(cross_val.mean() * 100) + "%")

# random forest classifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
pred_rf = rf.predict(x_test)
print(classification_report(y_test, pred_rf))
cross_val = cross_val_score(estimator = rf, X = x_train, y = y_train, cv = 10)
print("Random Forest Accuracy = " + str(cross_val.mean() * 100) + "%")

# decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
pred_dt = dt.predict(x_test)
print(classification_report(y_test, pred_dt))
cross_val = cross_val_score(estimator = dt, X = x_train, y = y_train, cv = 10)
print("Decision Tree Accuracy = " + str(cross_val.mean() * 100) + "%")

# stochastic gradient descent classifier
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
pred_sgd = sgd.predict(x_test)
print(classification_report(y_test, pred_sgd))
cross_val = cross_val_score(estimator = sgd, X = x_train, y = y_train, cv = 10)
print("SGD Accuracy = " + str(cross_val.mean() * 100) + "%")

# check if classifier file exists, and that it is not empty
if os.path.isfile("classifiers/white_wine_classifier") and (os.path.getsize("classifiers/white_wine_classifier") > 0):
    # load classifier from file
    ifile = bz2.BZ2File("classifiers/white_wine_classifier", "rb")
    rf_optimised = pickle.load(ifile)
    ifile.close()
else:
    # hyperparameter tuning for random forest classifier
    # number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 10)]
    # number of features to consider at every split
    max_features = ["sqrt"]
    # maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] + [None]
    # minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # method of selecting samples for training each tree
    bootstrap = [True, False]

    # create random grid
    random_grid = {"n_estimators": n_estimators,
                    "max_features": max_features,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "bootstrap": bootstrap}

    # random search of parameters, using 3 fold cross validation, search across 100 different combinations, and use all available cores
    rf_optimised = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose = 2, random_state = 42, n_jobs = -1)

# fit the random search model
rf_optimised.fit(x_train, y_train)
# print best parameters
pred_optimised = rf_optimised.predict(x_test)
print(classification_report(y_test, pred_optimised))
print(rf_optimised.best_params_)

# random forest classifier with optimised parameters
rfeval = cross_val_score(estimator = rf_optimised, X = x_train, y = y_train, cv = 10)
print("Optimised Random Forest Accuracy = " + str(rfeval.mean() * 100) + "%")

# save classifier to file
ofile = bz2.BZ2File("classifiers/white_wine_classifier", "wb")
pickle.dump(rf_optimised, ofile)
ofile.close()