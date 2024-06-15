import sys
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
sys.path.append("..")
from matplotlib import pyplot
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle

'''
#train = pd.read_csv("../Dataset/train_1_pp3.csv")
#val = pd.read_csv("../Dataset/val_pp3.csv")
#test = pd.read_csv("../Dataset/test_pp3.csv")
#train = pd.read_csv("../Dataset/train_1_pp3_MinMaxScaler.csv")
#val = pd.read_csv("../Dataset/val_pp3_MinMaxScaler.csv")
#test = pd.read_csv("../Dataset/test_pp3_MinMaxScaler.csv")
#TON_IoT_binary
#train = pd.read_csv('../Dataset/Ton_IoT/train_pp_binary.csv')
#val = pd.read_csv('../Dataset/Ton_IoT/val_pp_binary.csv')
#test = pd.read_csv('../Dataset/Ton_IoT/test_pp_binary.csv')
#Bot_IoT_binary
train = pd.read_csv('../Dataset/BoT-IoT/train_pp_binary.csv') # Bot_IoT_train_binary
val = pd.read_csv('../Dataset/BoT-IoT/val_pp_binary.csv') # Bot_IoT_val_binary
test = pd.read_csv('../Dataset/BoT-IoT/test_pp_binary.csv') # Bot_IoT_test_binary
#target = "label" #UNSW-NB15_binary
#target = "label" #TON_IoT_binary
target = "attack" #Bot_IoT_binary
x_train, y_train = train.drop([target], axis=1), train[target]
x_val, y_val = val.drop([target], axis=1), val[target]
x_test, y_test = test.drop([target], axis=1), test[target]
print("Dataset loaded")


print('y_train Shape: ', '\t', y_train.shape)
print('y_val Shape: ', '\t\t', y_val.shape)
print('y_test Shape: ', '\t\t', y_test.shape)
print("Original shape x_train: ", x_train.shape)
print("Original shape x_val: ", x_val.shape)
print("Original shape x_test: ", x_test.shape)
'''


"""
# Feature selection on Binary data whit XGBoost
def select_features(x_train, y_train, x_test, x_val):
	print("feature selection with XGBoost started...")
	# configured to select a subset of features whit XGBOoST
	# no Data flip threshold on 19 features is = 0.013172519
	# threshold used in paper=0.0037799259
    #0.0037799259
    # binary_TON_IoT_threshold = 0.0044252956
    # Binary_BOT_IoT_threshold = 0.054992266 # 7 features # used
	fs = SelectFromModel(XGBClassifier(),threshold=0.0037799259)
	#fs = SelectFromModel(GradientBoostingClassifier(),threshold=0.0041)#max_features=20)#threshold=0.006)#threshold=0.0041)#max_features=20)#0.006
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

"""
# Experimental trail with picle save
data = [x_train_fs, x_test_fs, x_val_fs, y_train, y_val, y_test]
file_path = 'Data_test/Binary_UNSW-NB15_fs_data.pickle' # UNSW-NB15
#file_path = 'Data_test/Binary_ton_fs_data.pickle' # TON
#file_path = 'Data_test/Binary_bot_fs_data.pickle' # BOT

#open the file in binary mode
with open(file_path, 'wb') as file:
	# serialize and write the variable
    pickle.dump(data, file)


print("The variable 'data' has been saved")
"""


#"""
# Experimental trail with picle load
#file_path = 'Data_test/Binary_UNSW-NB15_fs_data.pickle' # UNSW-NB15
#file_path = 'Data_test/Binary_ton_fs_data.pickle' # TON
file_path = 'Data_test/Binary_bot_fs_data.pickle' # BOT

#open the file in binary mode
with open(file_path, 'rb') as file:
	# Deserialize and retrieve the variable form the file
    x_train_fs, x_test_fs, x_val_fs, y_train, y_val, y_test = pickle.load(file)

print("The variable 'data' has been loaded successfully")
print("shape x_train_fs: ",x_train_fs.shape)
print("New shape x_val_fs: ",x_val_fs.shape)
print("New shape x_test_fs: ",x_test_fs.shape)
print('y_train Shape:','\t', y_train.shape)
print('y_val Shape:','\t', y_val.shape)
print('y_test Shape:','\t', y_test.shape)
#time.sleep(2)
#"""

import winsound
print("finish")
print("sound alert start... ")
duration = 3000 # miliseconds
freq = 400
winsound.Beep(freq, duration)


# Defining all the model to test the feature selection data set on

""""
# XGBoost model
model = XGBClassifier()
"""

"""
# Random Forest model
params = {"criterion": "entropy",
              "bootstrap":True,
              "n_estimators": 200, #the original test in my paper was wit 200
              #"n_estimators": 20, #test
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
model = DecisionTreeClassifier(max_depth=9) # testing 2 val ac = 92.036, 5 val ac=93.448 ,7 val ac =93.813 ,8 val ac= 94.146,9 val ac = 94.347
"""

"""
# Support Vector Machine (SVM) model
#model = svm.SVC(kernel='poly', degree=3, C=1)  
model = svm.SVC(kernel='rbf',gamma='scale', degree=3, C=1.12) # settings from related work 
"""

"""
# k-Nearest-Neighbour (KNN) model 
# 3,5,7,9,11 #9 =95.215, 93.946 # 5 val ac = 93.78 # 7 val ac = 93.79 #9 = val ac=93.94 #11 vl ac = 93.923
# diferent n_neighbora from related work 3,5,7,9,11 tested
model = KNeighborsClassifier(n_neighbors=3) # settings form related work
"""

"""
# logistic Regression (LR) model
#model = LogisticRegression(max_iter = 1000, C = 0.01, penalty = "l2",random_state=10)
#model = LogisticRegression(max_iter = 1000, C = 100, penalty = "l1", solver = "liblinear")
model = LogisticRegression(max_iter = 1000, random_state=10) #setings form related work used in paper
"""

#"""
#Artificaial Neural Network (ANN) model
# ANN whit sklearn
#model = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', learning_rate=0.02, max_iter=500)
# hidden layer sizes to test = 5,10,15,30,50,100, 150
model = MLPClassifier(hidden_layer_sizes=(150,),solver='adam',learning_rate='adaptive', learning_rate_init=0.02,) # settings form related work used in papr
#"""

"""
# Baseline test on model Before feature selection
start_time = time.time()
print("Model fitting starts...")
# Filing the model
model.fit(x_train, y_train)
print("The model Parameters on: \n",model)
# making prediction on train and test data
y_pred_train = model.predict(x_train)
y_pred_val = model.predict(x_val)
y_pred_test = model.predict(x_test)
"""

#"""
#feature selection dataset whit Xgboost feature selection
# fit the model whit feature selection dataset
start_time = time.time()
print("Model fitting starts...")
model.fit(x_train_fs, y_train)
# evaluate the model
print("The model Parameters on: \n",model)
y_pred_train = model.predict(x_train_fs)
y_pred_val = model.predict(x_val_fs)
y_pred_test = model.predict(x_test_fs)
#"""


print("Results on model model...")
elapsed_time = round((time.time() - start_time), 3)
print(f"Runtime: {elapsed_time}s")
# Same code for every model
# getting the accuracy
train_accuracy = round(accuracy_score(y_train, y_pred_train),5)
val_accuracy = round(accuracy_score(y_val, y_pred_val),5)
test_accuracy = round(accuracy_score(y_test, y_pred_test),5)
# balanced accuracy
train_accuracy_balanced = round(balanced_accuracy_score(y_train, y_pred_train),5)
val_accuracy_balanced = round(balanced_accuracy_score(y_val, y_pred_val),5)
test_accuracy_balanced = round(balanced_accuracy_score(y_test, y_pred_test),5)
# precision real and f1 on the test set
precision= round(precision_score(y_test, y_pred_test,average="weighted",zero_division=0), 5)
recall = round(metrics.recall_score(y_test,y_pred_test,average="weighted"), 5)
f1 = round(metrics.f1_score(y_test, y_pred_test,average="weighted"), 5)
# printin the accuracy and balanced accuracy
print(f"\nTraining accuracy: {train_accuracy*100}\nVal accuracy: \t {val_accuracy*100}\nTest accuracy: \t {test_accuracy*100}")
print(f"\nBalanced Training accuracy: {train_accuracy_balanced*100}\nBalanced Val accuracy: \t {val_accuracy_balanced*100}\nBalanced Test accuracy: \t {test_accuracy_balanced*100}\n")
print("precision: \t",precision)
print("recall:\t", recall)
print("f1:\t",f1,"\n")
# classification report on test data
print("Classification report: ")
print(metrics.classification_report(y_test, y_pred_test,zero_division=0))
# Create the confusion matrix on test predictions
cm = metrics.confusion_matrix(y_test, y_pred_test)
print("confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred_test))
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
pyplot.show()
