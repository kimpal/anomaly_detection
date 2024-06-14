# Code to run multiple models on different datasets and with gridsearch and set parameters
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
train = pd.read_csv(Bot_IoT_train)
val = pd.read_csv(Bot_IoT_val)
test = pd.read_csv(Bot_IoT_test)
#train = pd.read_csv(Bot_IoT_train_binary)
#val = pd.read_csv(Bot_IoT_val_binary)
#test = pd.read_csv(BoT_IoT_test_binary)
#train = pd.read_csv(UNSW_NB15_train_dataset)
#val = pd.read_csv(UNSW_NB15_val_dataset)
#test = pd.read_csv(UNSW_NB15_test_dataset)

# list of respective target values on different dataset
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
print("FeatureSelection with XGBoost starts...")
# Feature selection on Binary data with XGBoost
start_time_select_features = time.time()
def select_features(x_train_inner, y_train_inner, x_test_inner, x_val_inner):
	# configured to select a subset of features the XGBoost
	fs_inner = SelectFromModel(XGBClassifier(), threshold=0.026606342)
	# learn relationship from training data
	fs_inner.fit(x_train_inner, y_train_inner)
	# transform train input data
	x_train_fs_inner = fs_inner.transform(x_train_inner)
	# transform test input data
	x_test_fs_inner = fs_inner.transform(x_test_inner)
	# transform val input data
	x_val_fs_inner = fs_inner.transform(x_val_inner)
	return x_train_fs_inner, x_test_fs_inner, x_val_fs_inner, fs_inner

# feature selection calling of the function
x_train_fs, x_test_fs, x_val_fs, fs = select_features(x_train, y_train, x_test, x_val)
select_features_time = time.time() - start_time_select_features
print("New shape x_train: ",x_train_fs.shape)
print("New shape x_val: ",x_val_fs.shape)
print("New shape x_test: ",x_test_fs.shape)
print(f"feature selection time {select_features_time:.4f} seconds\n")
#"""


# defining the parameters and grid for the models

#XGBoostt model
XGBoost_model = XGBClassifier()
#GGBoost_param_grid = {}

#defing DT model
dt_model= DecisionTreeClassifier()
# Hyperparameter grid for Decision Tree
dt_param_grid = {'max_depth': [2,5,7,8,9]}

# SVM model
svm_model = svm.SVC(kernel='rbf',gamma='scale', degree=3, C=1.12)
# hyper parameter grid fo SVM
#svm_param_grid = {}

# ANN model
ann_model = MLPClassifier(max_iter=1000)
# Hyperparameter grid for Artificial Neural Network
ann_param_grid = {'hidden_layer_sizes': [(5,),(10,),(30,),(50,),(100,), (150,)]}

#KNN model
knn_model = KNeighborsClassifier()
# Hyperparameter grid for k-Nearest Neighbors
knn_param_grid = {'n_neighbors': [3,5,7,9,11]}

#Logistic regression
lr_model = LogisticRegression(max_iter = 2000, random_state=10)
# Hyperparameter grid logistic regression
#lr_param_grid = {}
# Random forest parameters
params = {
	"criterion": "entropy",
	"bootstrap": True,
	"n_estimators": 200,
	"max_depth": 50,
	"min_samples_split": 2,
	"min_samples_leaf": 1,
	"n_jobs": -1
}
# Create a RandomForestClassifier instance with the specified parameters
rf_model = RandomForestClassifier(**params)

# The models to evaluate is in the list
models_to_evaluate = [
	("XGBoost",XGBoost_model ,{}),
	("RandomForest", rf_model, {}),
	("DT", dt_model, dt_param_grid),
	("SVM",svm_model, {}),
	("KNN",knn_model, knn_param_grid),
	("LR",lr_model, {}),
	("ANN",ann_model, ann_param_grid),
]


print("\n------------------------- evaluation starting ----------------------------------------\n")
def print_metrics(dataset_name, y_true, y_pred):
	print(f"{dataset_name} Set:")
	print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
	print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred)}")
	print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0)}")
	print(f"Recall: {recall_score(y_true, y_pred, average='weighted')}")
	print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted')}")
	print("\nConfusion Matrix:")
	print(confusion_matrix(y_true, y_pred))
	print("\nClassification Report:")
	print(classification_report(y_true, y_pred, zero_division=0))
	print("\n")


def model_training_and_prediction(feature_set_name, model_name, model, param_grid, x_train_input, y_train_input, x_val_input, y_val_input, x_test_input, y_test_input):
	#printing model_name, dataset set name and shape
	print(f"Training {model_name} with hyperparameter tuning on the {feature_set_name} feature set...")
	print(f"x_train_{feature_set_name}_feature_set: {x_train_input.shape}, x_val_{feature_set_name}_feature_set: {x_val_input.shape}, x_test_{feature_set_name}_feature_set: {x_test_input.shape}")

	# Perform GridSearchCV for hyperparameter tuning
	grid_search = GridSearchCV(model, param_grid, cv=2, scoring='accuracy')
	grid_search.fit(x_train_input, y_train_input)

	# Get the best model
	best_model = grid_search.best_estimator_

	# Train the model
	start_time = time.time()
	best_model.fit(x_train_input, y_train_input)

	# Evaluate on the training se
	y_train_pred = best_model.predict(x_train_input)
	print_metrics(f"Training on {feature_set_name} feature set", y_train_input, y_train_pred)

	# Evaluate on validation set
	y_val_pred = best_model.predict(x_val_input)
	print_metrics(f"Validation on {feature_set_name} feature set", y_val_input, y_val_pred)

	# Evaluate on the test set
	y_test_pred = best_model.predict(x_test_input)
	print_metrics(f"Test on {feature_set_name} feature set", y_test_input, y_test_pred)
	training_time = time.time() - start_time

	# Print the best hyperparameters and training time
	print(f"Best hyperparameters for {model_name}: {grid_search.best_params_} if empty:",
		  '{} No grid param defined and model is trained on:\n', model)
	#print("the best model:\n", best_model) # maybe include
	print(f"Training time for {model_name} on the {feature_set_name} feature set: {training_time:.4f} seconds\n")


def evaluate_models(x_train_input=x_train, y_train_input=y_train, x_val_input=x_val, y_val_input=y_val, x_test_input=x_test, y_test_input=y_test, models:models_to_evaluate=None, x_train_fs_input=x_train_fs, x_val_fs_input=x_val_fs, x_test_fs_input=x_test_fs):
	# Record the start time
	total_start_time = time.time()

	if len(models) < 1:
		print("No models to evaluate.")
	else:
		for model_name, model, param_grid in models:
			#caling the model_training_and_prediction function that preform grid search, model training and evaluation on the original data set
			model_training_and_prediction('original', model_name, model, param_grid, x_train_input, y_train_input, x_val_input, y_val_input, x_test_input, y_test_input)

			# Check if reduced feature sets are provided; if not, use full feature sets
			if x_train_fs_input is None or x_val_fs_input is None or x_test_fs_input is None:
				print("Error: Reduced feature sets are not provided.")
				break

			# calling the model_training_and_prediction function that preforms grid search, model training and evaluation on the reduced feature set
			model_training_and_prediction('reduced', model_name, model, param_grid, x_train_fs_input, y_train_input, x_val_fs_input, y_val_input, x_test_fs_input, y_test_input)
			print("------------------------Next Model---------------------")

	# Calculate and print the total run time
	total_run_time = time.time() - total_start_time
	print(f"Total run time for all models training and evaluations: {total_run_time:.4f} seconds")


# Assuming x_train, y_train, x_val, y_val, x_test, y_test, x_train_fs, x_val_fs, x_test_fs is defined
#evaluate_models(x_train_input=x_train, y_train_input=y_train, x_val_input=x_val, y_val_input=y_val, x_test_input=x_test, y_test_input=y_test, models=models_to_evaluate, x_train_fs_input=x_train_fs, x_val_fs_input=x_val_fs, x_test_fs_input=x_test_fs):
evaluate_models(x_train, y_train, x_val, y_val, x_test, y_test, models_to_evaluate, x_train_fs, x_val_fs, x_test_fs)
