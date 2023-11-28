
# In[18]:
import sys
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
sys.path.append("..")
from matplotlib import pyplot
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
#from Functions.UNSW_DF import DF_XY, DF_preprocessed_traintest

train = pd.read_csv("../Dataset/train_1_pp3_multi.csv")
val = pd.read_csv("../Dataset/val_pp3_multi.csv")
test = pd.read_csv("../Dataset/test_pp3_multi.csv")
x_train, y_train = train.drop(["attack_cat"], axis=1), train["attack_cat"]
x_val, y_val = val.drop(["attack_cat"], axis=1), val["attack_cat"]
x_test, y_test = test.drop(["attack_cat"], axis=1), test["attack_cat"]
print("Dataset loaded")

"""
# specifically for ANN
x_train = np.asarray(x_train).astype(np.float64)
y_train = np.asarray(y_train).astype(np.float64)

x_val = np.asarray(x_val).astype(np.float64)
y_val = np.asarray(y_val).astype(np.float64)

x_test = np.asarray(x_test).astype(np.float64)
y_test = np.asarray(y_test).astype(np.float64)
"""

print('y_train Shape:','\t', y_train.shape)
print('y_val Shape:','\t', y_val.shape)
print('y_test Shape:','\t', y_test.shape)
print("Original shape x_train:",'\t', x_train.shape)
print("Original shape x_val:",'\t', x_val.shape)
print("Original shape x_test:",'\t', x_test.shape)

"""
print("FeatureSelection whit XGBoost starts...")
# Feature selection on Binary data whit XGBoost
def select_features(x_train, y_train, x_test, x_val):
	# configured to select a subset of features whit XGBOoST
    # threshold when data set is not flipped to get 19 features = 0.0064413943
	# thershold when MinMaxScaler =0.004771341
	#0.006284813
	fs = SelectFromModel(XGBClassifier(),threshold=0.00477)#threshold=0.006)#threshold=0.0075)#max_features=20)#0.006
	# learn relationship from training data
	fs.fit(x_train, y_train)
	# transform train input data
	x_train_fs = fs.transform(x_train)
	# transform test input data
	x_test_fs = fs.transform(x_test)
	# transform val input data
	x_val_fs = fs.transform(x_val)
	return x_train_fs, x_test_fs, x_val_fs, fs
# feature selection
x_train_fs, x_test_fs, x_val_fs, fs = select_features(x_train, y_train, x_test, x_val)
print("New shape x_train: ",x_train_fs.shape)
print("New shape x_val: ",x_val_fs.shape)
print("New shape x_test: ",x_test_fs.shape)
"""


# Defining all the model to test the feature selection data set on

#"""
# XGBoost model
model = XGBClassifier()
#"""

"""
# Random Frst model
params = {"criterion": "entropy",
              "bootstrap":True,
			  "n_estimators": 20, # new test
              #"n_estimators": 200,
              "max_depth": 50,
              "min_samples_split":2,
              "min_samples_leaf": 1,
              "n_jobs": -1}
# define the model
model = RandomForestClassifier(**params)
model.set_params(**params)
"""

"""
# Decision Tree model
#model = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20, min_samples_split=13, max_features=45 ,ccp_alpha=0.0)
#max depth to test 2,5,7,8,9
model = DecisionTreeClassifier(max_depth=5) # best identified 9 #testing 2 val_ac=64.495, 5 val_ac=71.699, 7 val_ac=77.05499999999999, 8 val_ac=78.123, 9 val_ac=78.593 # settings form related work
"""

"""
# Support Vector Machine (SVM) model
#model = svm.SVC(kernel='poly', degree=3, C=1)  
model = svm.SVC(kernel='rbf',gamma='scale', degree=3, C=1.12) # settings from related work
"""

"""
# k-Nearest-Neighbour (KNN) model
# diferent n_neighbora from related work 3,5,7,9,11
model = KNeighborsClassifier(n_neighbors=3,) # settings form related work # n_neighbors=3
"""

"""
# logistic Regression (LR) model
model = LogisticRegression(max_iter = 1000, C = 0.01, penalty = "l2",random_state=10)
#model = LogisticRegression(max_iter = 1000, random_state=10) #setings form related work
#model = LogisticRegression(max_iter = 2000, random_state=10) #adjusted max_iter to work whit standardScaler
"""

"""
#Artificaial Neural Network (ANN) model
# ANN whit sklearn
#model = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', learning_rate=0.02, max_iter=500)
#model = MLPClassifier(hidden_layer_sizes=(5,10,15,30,50,100),solver='adam',learning_rate='adaptive', learning_rate_init=0.02,) # settings form related work
model = MLPClassifier(hidden_layer_sizes=(5,),solver='adam',learning_rate='adaptive', learning_rate_init=0.02,) # settings form related work
"""

#"""
# Baseline test on model Before feature selection
start_time = time.time()
print("Model fitting start...")
# Fitting the model
model.fit(x_train, y_train)
print("The model Parameters on: \n",model)
# making prediction on train and test data
y_pred_train = model.predict(x_train)
y_pred_val = model.predict(x_val)
y_pred_test = model.predict(x_test)
#"""

"""
#feature selection dataset whit Xgboost
# fit the model whit feature selection dataset
start_time = time.time()
print("Model fitting start...")
# Fitting the model
model.fit(x_train_fs, y_train)
# evaluate the model
print("The model Parameters on: \n",model)
y_pred_train = model.predict(x_train_fs)
y_pred_val = model.predict(x_val_fs)
y_pred_test = model.predict(x_test_fs)
"""

#"""
print("Results on model...")
elapsed_time = round((time.time() - start_time), 3)
print(f"Runtime: {elapsed_time}s")
# Same code for every model
# getting the accuracy
train_accuracy = round(accuracy_score(y_train, y_pred_train),5)
val_accuracy = round(accuracy_score(y_val, y_pred_val),5)
test_accuracy = round(accuracy_score(y_test, y_pred_test),5)
print(f"Training accuracy: {train_accuracy*100}\nVal accuracy: \t {val_accuracy*100}\nTest accuracy: \t {test_accuracy*100}")
# classification report on test data
print(metrics.classification_report(y_test, y_pred_test, zero_division=0))
# Create the confusion matrix on test predictions
cm = metrics.confusion_matrix(y_test, y_pred_test)
print("confusion matrix:")
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
pyplot.show()
#"""
