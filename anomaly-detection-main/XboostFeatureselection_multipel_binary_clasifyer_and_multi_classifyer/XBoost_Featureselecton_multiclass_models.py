
# In[18]:
import os
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
from sklearn.metrics import accuracy_score, precision_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import tracemalloc
import pickle
#"""
#train = pd.read_csv("../Dataset/train_1_pp3_multi.csv")
#val = pd.read_csv("../Dataset/val_pp3_multi.csv")
#test = pd.read_csv("../Dataset/test_pp3_multi.csv")
#train = pd.read_csv("../Dataset/train_1_pp3_multi_MinMaxScaler.csv")
#val = pd.read_csv("../Dataset/val_pp3_multi_MinMaxScaler.csv")
#test = pd.read_csv("../Dataset/test_pp3_multi_MinMaxScalr.csv")
train = pd.read_csv('../Dataset/Ton_IoT/train_pp_multi.csv')
val = pd.read_csv('../Dataset/Ton_IoT/val_pp_multi.csv')
test = pd.read_csv('../Dataset/Ton_IoT/test_pp_multi.csv')
#train = pd.read_csv('../Dataset/BoT-IoT/train_pp_multi.csv')
#val = pd.read_csv('../Dataset/BoT-IoT/val_pp_multi.csv')
#test = pd.read_csv('../Dataset/BoT-IoT/test_pp_multi.csv')
#target = "attack_cat" #UNSW-NB15
target = "type" #TON_IoT
#target = "category" # BoT-IoT
x_train, y_train = train.drop([target], axis=1), train[target]
x_val, y_val = val.drop([target], axis=1), val[target]
x_test, y_test = test.drop([target], axis=1), test[target]
print("Dataset loaded")


print('y_train Shape:','\t', y_train.shape)
print('y_val Shape:','\t', y_val.shape)
print('y_test Shape:','\t', y_test.shape)
print("Original shape x_train:",'\t', x_train.shape)
print("Original shape x_val:",'\t', x_val.shape)
print("Original shape x_test:",'\t', x_test.shape)
#"""

"""
# Feature selection on Binary data whit XGBoost
def select_features(x_train, y_train, x_test, x_val):
    print("FeatureSelection whit XGBoost starts...")
    # configured to select a subset of features the XGBoost
    # threshold when data set is not flipped to get 19 features = 0.0064413943
    # threshold when MinMaxScaler =0.004771341
    #Threshold used in paper 0.007539547
    # Bot_IoT_Threshold = 0.026606342 # 7 features # used
    # TON_IOT_Threshold = 0.0075270594
    #0.0069906875
    fs = SelectFromModel(XGBClassifier(),threshold=0.026606342)
    # learn relationship from training data
    fs.fit(x_train, y_train)
    # transform train input data
    x_train_fs = fs.transform(x_train)
    # transform test input data
    x_test_fs = fs.transform(x_test)
    # transform val input data
    x_val_fs = fs.transform(x_val)
    return x_train_fs, x_test_fs, x_val_fs, fs
# feature selection function cal
x_train_fs, x_test_fs, x_val_fs, fs = select_features(x_train, y_train, x_test, x_val)
print("New shape x_train: ",x_train_fs.shape)
print("New shape x_val: ",x_val_fs.shape)
print("New shape x_test: ",x_test_fs.shape)
"""

"""
# Experimental trail with picle save
data = [x_train_fs, x_test_fs, x_val_fs, y_train, y_val, y_test]
#file_path = 'Data_test/ton_fs_data.pickle' # TON
file_path = 'Data_test/bot_fs_data.pickle' # BOT

#open the file in binary mode
with open(file_path, 'wb') as file:
    # serialize and write the variable
    pickle.dump(data, file)


print("The variable 'data' has been saved")
"""


"""
# Experimental trail with picle load
file_path = 'Data_test/ton_fs_data.pickle' # TON
#file_path = 'Data_test/bot_fs_data.pickle' # BOT

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
time.sleep(5)
"""



# Defining all the model to test the feature selection data set on

"""
# XGBoost model
model = XGBClassifier()
"""

"""
# Random Frst model
params = {"criterion": "entropy",
              "bootstrap":True,
              #"n_estimators": 20, # new test
              "n_estimators": 200,
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
#max depth to test
#max depth to test 2,5,7,8,9
model = DecisionTreeClassifier(max_depth=9) # best identified 9 #testing 2 val_ac=64.495, 5 val_ac=71.699, 7 val_ac=77.05499999999999, 8 val_ac=78.123, 9 val_ac=78.593 # settings form related work
"""

"""
# Support Vector Machine (SVM) model
#model = svm.SVC(kernel='poly', degree=3, C=1)  
model = svm.SVC(kernel='rbf',gamma='scale', degree=3, C=1.12) # settings from related work used in paper
"""

"""
# k-Nearest-Neighbour (KNN) model
# different n_neighbora from related work 3,5,7,9,11 tested
model = KNeighborsClassifier(n_neighbors=3,) # settings form related work #
"""

"""
# logistic Regression (LR) model
#model = LogisticRegression(max_iter = 1000, C = 0.01, penalty = "l2",random_state=10)
model = LogisticRegression(max_iter = 1000, random_state=10) #setings form related work and used in paper
#model = LogisticRegression(max_iter = 2000, random_state=10) #adjusted max_iter to work whit standardScaler just a test
"""

#"""
#Artificaial Neural Network (ANN) model
# ANN whit sklearn
#model = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', learning_rate=0.02, max_iter=500)
#model = MLPClassifier(hidden_layer_sizes=(5,10,15,30,50,100),solver='adam',learning_rate='adaptive', learning_rate_init=0.02,) # settings form related work
# hidden_layer_sizes tested: 5,10,15,30,50,100, 150
model = MLPClassifier(hidden_layer_sizes=(150,),solver='adam',learning_rate='adaptive', learning_rate_init=0.02,) # settings form related work used in paper
#"""

# starting the monitoring
tracemalloc.start()
prev_time = os.times().user

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

# displaying the memory
print("memory usage: ",tracemalloc.get_traced_memory())
# stopping the library
tracemalloc.stop()
# stopping the cpu time measuring
current_time = os.times().user
cpu_percent = (current_time - prev_time) * 100 / os.cpu_count()
print(f"CPU Usage: {cpu_percent}%")

#"""
print("Results on model...")
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
# precision recall and f1 on the test set
precision= round(metrics.precision_score(y_test, y_pred_test,average="weighted", zero_division=0), 2)
recall = round(metrics.recall_score(y_test,y_pred_test,average="weighted"), 2)
f1 = round(metrics.f1_score(y_test, y_pred_test,average="weighted"), 2)
#Printin all the evaluation scores
print(f"Training accuracy: {train_accuracy*100}\nVal accuracy: \t {val_accuracy*100}\nTest accuracy: \t {test_accuracy*100}")
print(f"\nBalanced Training accuracy: {train_accuracy_balanced*100}\nBalanced Val accuracy: \t {val_accuracy_balanced*100}\nBalanced Test accuracy: \t {test_accuracy_balanced*100}\n")
print("precision:\t",precision)
print("recall:\t", recall)
print("f1:\t",f1,"\n")
# classification report on test data
print("Classification report: ")
print(metrics.classification_report(y_test, y_pred_test, zero_division=0))
# Create the confusion matrix on test predictions
cm = metrics.confusion_matrix(y_test, y_pred_test)
print("confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred_test))
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
pyplot.show()
#"""
