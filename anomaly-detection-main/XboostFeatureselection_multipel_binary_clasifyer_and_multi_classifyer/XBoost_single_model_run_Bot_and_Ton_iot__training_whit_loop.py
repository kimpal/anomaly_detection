# Code This code is adapted to run only SVM
import sys
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
sys.path.append("..")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, balanced_accuracy_score, f1_score, recall_score, \
	confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")
# datasets defined for multiclass and binary
#UNSW_NB15_train_dataset, UNSW_NB15_val_dataset, UNSW_NB15_test_dataset = "../Dataset/train_1_pp3_multi.csv", "../Dataset/val_pp3_multi.csv", "../Dataset/test_pp3_multi.csv"
#UNSW_NB15_train_binary_dataset, UNSW_NB15_val_binary_dataset, UNSW_NB15_test_binary_dataset = "../Dataset/train_1_pp3_binary.csv", "../Dataset/val_pp3_binary.csv", "../Dataset/test_pp3_binary.csv"
#TON_IoT_train, TON_IoT_val, TON_IoT_test = '../Dataset/Ton_IoT/train_pp_multi.csv', '../Dataset/Ton_IoT/val_pp_multi.csv', '../Dataset/Ton_IoT/test_pp_multi.csv'
#TON_IoT_train_binary, TON_IoT_val_binary, TON_IoT_test_binary = '../Dataset/Ton_IoT/train_pp_binary.csv', '../Dataset/Ton_IoT/val_pp_binary.csv', '../Dataset/Ton_IoT/test_pp_binary.csv'
Bot_IoT_train, Bot_IoT_val, Bot_IoT_test = '../Dataset/BoT-IoT/train_pp_multi.csv','../Dataset/BoT-IoT/val_pp_multi.csv', '../Dataset/BoT-IoT/test_pp_multi.csv'
#Bot_IoT_train_binary, Bot_IoT_val_binary, Bot_IoT_test_binary = '../Dataset/BoT-IoT/train_pp_binary.csv','../Dataset/BoT-IoT/val_pp_binary.csv', '../Dataset/BoT-IoT/test_pp_binary.csv'

#train = pd.read_csv(TON_IoT_train)
#val = pd.read_csv(TON_IoT_val)
#test = pd.read_csv(TON_IoT_test)
#train = pd.read_csv(TON_IoT_train_binary)
#val = pd.read_csv(TON_IoT_val_binary)
#test = pd.read_csv(TON_IoT_test_binary)
train = pd.read_csv(Bot_IoT_train) # Left To run on the ful feature set
val = pd.read_csv(Bot_IoT_val) #  # Left To run on the ful feature set
test = pd.read_csv(Bot_IoT_test) # # Left To run on the ful feature set
#train = pd.read_csv(Bot_IoT_train_binary)
#val = pd.read_csv(Bot_IoT_val_binary)
#test = pd.read_csv(Bot_IoT_test_binary)
#train = pd.read_csv(UNSW_NB15_train_dataset)
#val = pd.read_csv(UNSW_NB15_val_dataset)
#test = pd.read_csv(UNSW_NB15_test_dataset)

# list of respective target values on different dataset dataset
#target_value = "attack_cat" # target values on Multiclass UNSW_NB15
#target_value = "label" # target values on Binary UNSW_NB15
#target_value = "type" # target values on Multiclass TON_IoT
#target_value = "label" # target values on Binary TON_IoT
target_value = "category" # target values on Multiclass Bot-IoT
#target_value = "attack" # target values on Binary Bot-IoT
x_train, y_train = train.drop([target_value], axis=1), train[target_value]
x_val, y_val = val.drop([target_value], axis=1), val[target_value]
x_test, y_test = test.drop([target_value], axis=1), test[target_value]
print("Dataset loaded")


print('y_train Shape:','\t', y_train.shape)
print('y_val Shape:','\t', y_val.shape)
print('y_test Shape:','\t', y_test.shape)
print("Original shape x_train:",'\t', x_train.shape)
print("Original shape x_val:",'\t', x_val.shape)
print("Original shape x_test:",'\t', x_test.shape)

#binary_TON_IoT_threshold = 0.0044252956
#multiclass_TON_IoT_threshold = 0.007512621
#Multiclass_BOT_IoT_threshold = 0.026606342 # 7 features # used
#binary_BOT_IoT_threshold = 0.03815346 #11 features since the original is 16
#Binary_BOT_IoT_threshold = 0.03753139 # 12 features
#Binary_BOT_IoT_threshold = 0.042484876 # 10 features
#Binary_BOT_IoT_threshold = 0.050731182 # 8 features
#Binary_BOT_IoT_threshold = 0.054992266 # 7 features # used
#0.026790911
#0.061215572
#"""
#"""
print("FeatureSelection whit XGBoost starts...")
# Feature selection on Binary data whit XGBoost
def select_features(x_train, y_train, x_test, x_val):
	# configured to select a subset of features the XGBoost
    # threshold when data set is not flipped to get 19 features = 0.0064413943
	# threshold when MinMaxScaler =0.004771341
	#Threshold used in paper 0.007539547
	# Bot_IoT_Threshold =
	# TON_IOT_Threshold = 0.0075270594
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
# feature selection
x_train_fs, x_test_fs, x_val_fs, fs = select_features(x_train, y_train, x_test, x_val)
print("New shape x_train: ",x_train_fs.shape)
print("New shape x_val: ",x_val_fs.shape)
print("New shape x_test: ",x_test_fs.shape)
#"""

#"""
# Support Vector Machine (SVM) model
#model = svm.SVC(kernel='poly', degree=3, C=1)
model = svm.SVC(kernel='rbf',gamma='scale', degree=3, C=1.12) # settings from related work used in paper
#"""
# DT_model
#model= DecisionTreeClassifier(max_depth=9)

"""
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
"""
# for loup to make the code run multiple times
for i in range(1):
	#"""
	print("Run",i+1)
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
	#"""

	#"""
	print("Results on model...")
	elapsed_time = round((time.time() - start_time), 3)
	print(f"Runtime: {elapsed_time}s")
	# Same code for every model
	# getting the accuracy
	train_accuracy = accuracy_score(y_train, y_pred_train)
	val_accuracy = accuracy_score(y_val, y_pred_val)
	test_accuracy = accuracy_score(y_test, y_pred_test)
	# balanced accuracy
	train_accuracy_balanced = balanced_accuracy_score(y_train, y_pred_train)
	val_accuracy_balanced = balanced_accuracy_score(y_val, y_pred_val)
	test_accuracy_balanced = balanced_accuracy_score(y_test, y_pred_test)
	# precision recall and f1 on the test set
	precision= precision_score(y_test, y_pred_test,average="weighted", zero_division=0)
	recall = recall_score(y_test,y_pred_test,average="weighted")
	f1 = f1_score(y_test, y_pred_test,average="weighted")
	#Printin all the evaluation scores
	print(f"Training accuracy: {train_accuracy}\nVal accuracy: \t {val_accuracy}\nTest accuracy: \t {test_accuracy}")
	print(f"\nBalanced Training accuracy: {train_accuracy_balanced}\nBalanced Val accuracy: \t {val_accuracy_balanced}\nBalanced Test accuracy: \t {test_accuracy_balanced}\n")
	print("precision:\t",precision)
	print("recall:\t", recall)
	print("f1:\t",f1,"\n")
	# classification report on test data
	print("Classification report: ")
	print(classification_report(y_test, y_pred_test, zero_division=0))
	# Create the confusion matrix on test predictions
	cm = confusion_matrix(y_test, y_pred_test)
	print("confusion matrix:")
	print(confusion_matrix(y_test, y_pred_test))
	#"""
