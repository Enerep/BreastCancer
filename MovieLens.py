import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Reading the Data
df = pd.read_csv("BreastCancer.csv")

# print(rate.head())

attributes = ["id", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
              "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
              "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
              "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
              "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
              "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]

df['diagnosis'] = df['diagnosis'] == 'M'

X = df[attributes].values
y = df['diagnosis'].values


print(df.head())
# Using Only half of the DataSet

print("Shape Of Whole DataSet: ", X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print("Decision Tree: ")
# Decision Tree training
dtmodel = DecisionTreeClassifier()

dtmodel.fit(X_train, y_train)


score = dtmodel.score(X_test, y_test)
print("Score of DT Training: ", score)
dt_y_pred = dtmodel.predict(X_test)

print("Precision Score: ", precision_score(y_test, dt_y_pred))
print("Recall Score: ", recall_score(y_test, dt_y_pred))

print("Logistic Regression: ")
# Logistic Regression training
lgmodel = LogisticRegression()

lgmodel.fit(X_train, y_train)


score = lgmodel.score(X_test, y_test)
print("Score of LG Training: ", score)
lg_y_pred = lgmodel.predict(X_test)


