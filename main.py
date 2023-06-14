import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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