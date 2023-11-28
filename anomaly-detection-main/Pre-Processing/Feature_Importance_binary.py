"""
Getting feature importance for multiclass classification
with: XGBoost, Random forest, Decision Tree and KNN
"""
# required  imports
import os
import shap
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib import pyplot
import category_encoders as ce
import matplotlib.pyplot as plt
from numpy import loadtxt, sort
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance

# getting the dataset file
main_path ="../Dataset/"
train = pd.read_csv("../Dataset/train_1_pp3.csv")
val = pd.read_csv("../Dataset/val_pp3.csv")
test = pd.read_csv("../Dataset/test_pp3.csv")
print('dataset in shape of train: ', train.shape)
print('dataset in shape of tes: ', test.shape)


# split data into X an Y multiclass
x_train, y_train = train.drop(['label'], axis=1), train['label']
x_test, y_test = test.drop(['label'], axis=1), test['label']

# split data into x an y on binary
#x_train, y_train = train.drop(["label"], axis=1), train["label"]
#x_test, y_test = test.drop(["label"], axis=1), test["label"]

# cod for models :https://machinelearningmastery.com/calculate-feature-importance-with-python/

"""
XGBoost
code for the XGBoost feature selection:
 https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
"""
#"""
# fit model on training data
XGBoost_model = XGBClassifier()
XGBoost_model.fit(x_train, y_train)
# get the feature importance
feature_importanceXGBC_scores = pd.Series(XGBoost_model.feature_importances_, index=x_train.columns).sort_values(ascending=False)
# saving the feature importance of XGBoost
pd.DataFrame({"Featureimportance list": feature_importanceXGBC_scores}).to_csv('no_data_flip_feature_importance_XGBoost_binary_val_split.csv')
# Plotting the feature importance
#fig, ax = plt.subplots(figsize=(10,10))
#xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax,importance_type='gain')
#plt.savefig('XGBClassifier_feature_importance.png')
# only plot the customized number of features
top_n = 40
plt.figure(figsize=(12, 16))
plt.title('Feature Importance')
feature_importanceXGBC_scores[:top_n].plot.barh();
plt.xlabel('Relative Importance')
plt.savefig('no_data_flip_XGBClassifier_feature_importance_binary_val_split.png')
plt.show()
#plt.show()
#explainer = shap.TreeExplainer(model_DT)
#shap_values = explainer.shap_values(x_train)
#shap.summary_plot(shap_values, x_train, plot_type="bar", show=False)
#shap_values = shap.TreeExplainer(XGBoost_model).shap_values(x_train)
#shap.summary_plot(shap_values(x_train) , plot_type="bar", show=False)
#plt.savefig('XGBOOST_test_label_encoder_al_cat.png')
#"""

"""
GradientBoost feature importance
"""
"""
# fit model on training data
GradientBoost_model = GradientBoostingClassifier()
GradientBoost_model.fit(x_train, y_train)
# get the feature importance
feature_importanceGradientBoost_scores = pd.Series(GradientBoost_model.feature_importances_, index=x_train.columns).sort_values(ascending=False)
# saving the feature importance of XGBoost
pd.DataFrame({"Featureimportance list": feature_importanceGradientBoost_scores}).to_csv('no_data_flip_feature_importance_GradientBoost_binary_val_split.csv')
# Plotting the feature importance
#fig, ax = plt.subplots(figsize=(10,10))
#xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax,importance_type='gain')
#plt.savefig('XGBClassifier_feature_importance.png')
# only plot the customized number of features
top_n = 40
plt.figure(figsize=(12, 16))
plt.title('Feature Importance')
feature_importanceGradientBoost_scores[:top_n].plot.barh();
plt.xlabel('Relative Importance')
plt.savefig('no_data_flip_GradientBoostClassifier_feature_importance_binary_val_split.png')
plt.show()
"""


"""
Random forest feature importance
"""
"""
# random forest model
RF_model = RandomForestClassifier(n_estimators=100, random_state=0)
# fit the model to training dataset
RF_model.fit(x_train, y_train)
# view feature scores
importanceRF = pd.Series(RF_model.feature_importances_, index=x_train.columns).sort_values(ascending=False)
print(importanceRF)
# export importanceRF to a document
pd.DataFrame({"Featureimportance list": importanceRF}).to_csv('feature_importance_RF_label_encoder_al_cat.csv')
# Plotting the feature importance
# only plot the customized number of features
top_n = 40
plt.figure(figsize=(12, 16))
plt.title('Feature Importance')
importanceRF[:top_n].plot.barh();
plt.xlabel('Relative Importance')
plt.savefig('RFClassifier_feature_importance_label_encoder_al_cat.png')
plt.show()
#shap_values = shap.TreeExplainer(RF_model).shap_values(x_train)
#shap.summary_plot(shap_values, x_train, plot_type="bar", show=False)
#plt.savefig('RF_test.png')
"""


"""
Decision Tree feature importance
cod for models :https://machinelearningmastery.com/calculate-feature-importance-with-python/
"""
"""
# decision tree for feature importance on a classification problem
# define the model
model_DT = DecisionTreeClassifier()
# fit the model
model_DT.fit(x_train, y_train)
# get importance
importanceDT = pd.Series(model_DT.feature_importances_, index=x_train.columns).sort_values(ascending=False)
# saving the feature importanceDT to a csv file
pd.DataFrame({"Featureimportance list": importanceDT}).to_csv('feature_importance_DT_label_encoder_al_cat.csv')
# Displaying the feature importance
print(importanceDT)
# plotting feature importance
top_n = 40
plt.figure(figsize=(12, 16))
plt.title('Feature Importance')
importanceDT[:top_n].plot.barh();
plt.xlabel('Relative Importance')
plt.savefig('feature_importance_DT_label_encoder_al_cat.png')
plt.show()
# plotting whit shap
#explainer = shap.TreeExplainer(model_DT)
#shap_values = explainer.shap_values(x_train)
#shap.summary_plot(shap_values, x_train, plot_type="bar", show=False)
#plt.savefig('DT_test.png')
"""


"""
KNeighborsClassifier 
Permutation Feature Importance
cod for models :https://machinelearningmastery.com/calculate-feature-importance-with-python/
"""
#neead to future investigate since
# permutation feature importance with knn for classification
"""
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_classes=4, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = KNeighborsClassifier()
# fit the model
model.fit(X, y)
# perform permutation importance
results = permutation_importance(model, X, y, scoring='accuracy')
# get importance
importance = pd.Series(results.importances_mean).sort_values(ascending=False)
#importance = pd.Series(results.importances_mean, index=X.column).sort_values(ascending=False)
pd.DataFrame({"Featureimportance list": importance}).to_csv('feature_importance_KNN.csv')
# summarize feature importance
print(importance)
#for i,v in enumerate(importance):
#	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
#plt.bar([x for x in range(len(importance))], importance)
#plt.show()
# plot feature importance
top_n = 40
plt.figure(figsize=(12, 16))
plt.title('Feature Importance')
importance[:top_n].plot.barh();
plt.xlabel('Relative Importance')
plt.savefig('feature_importance_KN.png')
#plt.savefig('KN_test.png')
#pyplot.bar([x for x in range(len(importanceKN))], importanceKN)
plt.show()
"""
"""
# define dataset
x_train, y_train = make_classification(n_samples=1000, n_classes=4, n_features=20, n_informative=5, n_redundant=5, random_state=1)
# permutation feature importance with knn for classification
# define the model
KN_model = KNeighborsClassifier()
# fit the model
KN_model.fit(x_train, y_train)
# perform permutation importance
#results = permutation_importance(KN_model, x_train, y_train, scoring='accuracy')
# get importance
#importanceKN = results.importances_mean
importanceKN = pd.Series(KN_model.feature_importances_, index=x_train.columns).sort_values(ascending=False)
# saving the feature importanceKN to a csv file
#pd.DataFrame({"Featureimportance list": importanceKN}).to_csv('feature_importance_KN.csv')
# summarize feature importance
for i,v in enumerate(importanceKN):
	print('Feature: %0d, Score: %.5f' % (i,v))
   
# plot feature importance
top_n = 40
plt.figure(figsize=(12, 16))
plt.title('Feature Importance')
importanceKN[:top_n].plot.barh();
plt.xlabel('Relative Importance')
plt.savefig('feature_importance_KN.png')
#plt.savefig('KN_test.png')
#pyplot.bar([x for x in range(len(importanceKN))], importanceKN)
plt.show()
"""
