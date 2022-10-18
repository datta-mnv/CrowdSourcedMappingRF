#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 11:15:11 2021

@author: plucky
"""
# Imported required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.linalg import eig
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import timeit

# Read the datasets from filepath
training_set = pd.read_csv(
    "/Users/plucky/Desktop/MSDS/6350/hw/HW-comp/Crowdsourced_Mapping/training.csv")

test_set = pd.read_csv(
    "/Users/plucky/Desktop/MSDS/6350/hw/HW-comp/Crowdsourced_Mapping/testing.csv")

training_set["class"].unique()

# Divide the datasets according to classes respectively
water_set = training_set[training_set["class"] == "water"]
forest_set = training_set[training_set["class"] == "forest"]
impervious_set = training_set[training_set["class"] == "impervious"]
farm_set = training_set[training_set["class"] == "farm"]
grass_set = training_set[training_set["class"] == "grass"]
orchard_set = training_set[training_set["class"] == "orchard"]


water_feature = water_set.drop(columns=["class"])
water_class = water_set["class"]
forest_feature = forest_set.drop(columns=["class"])
forest_class = forest_set["class"]
impervious_feature = impervious_set.drop(columns=["class"])
impervious_class = impervious_set["class"]
farm_feature = farm_set.drop(columns=["class"])
farm_class = farm_set["class"]
grass_feature = grass_set.drop(columns=["class"])
grass_class = grass_set["class"]
orchard_feature = orchard_set.drop(columns=["class"])
orchard_class = orchard_set["class"]

x_water_train, x_water_test, y_water_train, y_water_test = train_test_split(
    water_feature, water_class, test_size=0.2)
x_forest_train, x_forest_test, y_forest_train, y_forest_test = train_test_split(
    forest_feature, forest_class, test_size=0.2)
x_impervious_train, x_impervious_test, y_impervious_train, y_impervious_test = train_test_split(
    impervious_feature, impervious_class, test_size=0.2)
x_farm_train, x_farm_test, y_farm_train, y_farm_test = train_test_split(
    farm_feature, farm_class, test_size=0.2)
x_grass_train, x_grass_test, y_grass_train, y_grass_test = train_test_split(
    grass_feature, grass_class, test_size=0.2)
x_orchard_train, x_orchard_test, y_orchard_train, y_orchard_test = train_test_split(
    orchard_feature, orchard_class, test_size=0.2)



x_training = pd.concat([x_farm_train, x_forest_train,
                       x_impervious_train])
x_testing = pd.concat([x_farm_test, x_forest_test,
                      x_impervious_test ])

y_training = pd.concat([y_farm_train, y_forest_train,
                       y_impervious_train ])
y_testing = pd.concat([y_farm_test, y_forest_test,
                      y_impervious_test ])


sm = SMOTE(random_state=42)
X_train, Y_train = sm.fit_resample(x_training, y_training)
X_test, Y_test = sm.fit_resample(x_testing, y_testing)


time_start = timeit.default_timer()

model = RandomForestClassifier(n_estimators=400, max_features=5,oob_score=True)
model.fit(X_train, Y_train)

y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

accuracy_score_test = metrics.accuracy_score(Y_test, y_test_pred)
accuracy_score_train = metrics.accuracy_score(
    Y_train, y_train_pred)

confusion_mat_test = confusion_matrix(Y_test, y_test_pred)
confusion_mat_train = confusion_matrix(Y_train, y_train_pred)

time_elapsed = (timeit.default_timer() - time_start)



feature_Set = training_set.drop(columns=["class"])
classification_set = training_set["class"]

SDATA = StandardScaler().fit_transform(feature_Set)
SDATA = pd.DataFrame(SDATA)

feature_correlation = feature_Set.corr().abs()

feature_eigen_values, feature_eigen_vectors = eig(feature_correlation)

feature_eigen_vectors = pd.DataFrame(feature_eigen_vectors)

m_for_eigen = list(range(0, 28))

L = np.sort(feature_eigen_values)[::-1]

idx = np.argsort(-1)

# Computation of PEV for each case m

PEV = []

for i in range(28):
    a = sum(L[0:i]/28)
    PEV.append(a)

# PEV(m) vs m

sns.set()
plt.plot(m_for_eigen, PEV)
plt.hlines(y=0.9, xmin=0, xmax=20, color='r', linestyle='--', linewidth=2)
plt.vlines(x=20, ymin=0, ymax=0.9, color='r', linestyle='--', linewidth=2)
plt.xlabel("Index of Eigen values")
plt.ylabel("Eigen values in decreasing order")
plt.title('Eigen values')
plt.title("PEV")
plt.show()

# find least r such that PEV(r)>90%
r = []
for i in range(28):
    if sum(L[0:i]) > 0.9*28:
        r.append(i)

print(r)

feature_eigen_vectors_r = feature_eigen_vectors.iloc[:, :20]

ZDATA = np.matmul(SDATA, feature_eigen_vectors_r)


forest_z_set = ZDATA[classification_set.values == "forest"]
farm_z_set = ZDATA[classification_set.values == "farm"]
impervious_z_set = ZDATA[classification_set.values == "impervious"]

forest_z_class = classification_set[classification_set.values == "forest"]
farm_z_class = classification_set[classification_set.values == "farm"]
impervious_z_class = classification_set[classification_set.values == "impervious"]


x_forest_z_train, x_forest_z_test, y_forest_z_train, y_forest_z_test = train_test_split(
    forest_z_set, forest_z_class, test_size=0.2)
x_impervious_z_train, x_impervious_z_test, y_impervious_z_train, y_impervious_z_test = train_test_split(
    farm_z_set, farm_z_class, test_size=0.2)
x_farm_z_train, x_farm_z_test, y_farm_z_train, y_farm_z_test = train_test_split(
    impervious_z_set, impervious_z_class, test_size=0.2)

x_z_training = pd.concat([ x_forest_z_train,
                          x_impervious_z_train, x_farm_z_train])
x_z_testing = pd.concat([ x_forest_z_test,
                         x_impervious_z_test, x_farm_z_test ])

y_z_training = pd.concat([ y_forest_z_train,
                          y_impervious_z_train, y_farm_z_train])
y_z_testing = pd.concat([ y_forest_z_test,
                         y_impervious_z_test, y_farm_z_test])


sm = SMOTE(random_state=42)
X_z_train, Y_z_train = sm.fit_resample(x_z_training, y_z_training)
X_z_test, Y_z_test = sm.fit_resample(x_z_testing, y_z_testing)


time_start = timeit.default_timer()

model_z = RandomForestClassifier(n_estimators=400, max_features=5,oob_score=True) # n_estimators = 100,200,300,400,500
model_z.fit(X_z_train, Y_z_train)

y_test_pred_z = model_z.predict(X_z_test)
y_train_pred_z = model_z.predict(X_z_train)

accuracy_score_test_z = metrics.accuracy_score(Y_z_test, y_test_pred_z)
accuracy_score_train_z = metrics.accuracy_score(
    Y_z_train, y_train_pred_z)

confusion_mat_test_z = confusion_matrix(Y_z_test, y_test_pred_z)
confusion_mat_train_z = confusion_matrix(Y_z_train, y_train_pred_z)

time_elapsed_z = (timeit.default_timer() - time_start)



test_class = test_set["class"]
test_features = test_set.drop(columns=["class"])

model_sam = RandomForestClassifier(n_estimators=1000, max_features=5)
model_sam.fit(X_train, Y_train)

y_test_pred = model_sam.predict(test_features)

accuracy_score_test = metrics.accuracy_score(test_class, y_test_pred)

confusion_mat_test = confusion_matrix(test_class, y_test_pred)


train_class = training_set["class"]
train_features = training_set.drop(columns=["class"])

y_train_pred = model.predict(train_features)

accuracy_score_train = metrics.accuracy_score(train_class, y_train_pred)

confusion_mat_train = confusion_matrix(train_class, y_train_pred)


# Feature imporatance for each feature

model.feature_importances_

# Sorted features importance

model_features_sorted = model.feature_importances_.argsort()

feature_names = test_features.columns.values


# Vertical plot for feature importances
fig = plt.figure()
fig.set_figwidth(6)
fig.set_figheight(10)
plt.hlines(y=(feature_names[model_features_sorted]), xmin=0, xmax=(
    model.feature_importances_[model_features_sorted]))
plt.plot((model.feature_importances_[
         model_features_sorted]), (feature_names[model_features_sorted]), 'D', markersize=10, markeredgecolor="orange", markeredgewidth=2)
plt.title("Features Imporatance for RF*", loc='center')
plt.xlabel('Imporatnce values')
plt.ylabel('Features')
plt.show()




















