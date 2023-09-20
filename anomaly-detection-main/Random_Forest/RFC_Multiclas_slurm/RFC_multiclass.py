#!/usr/bin/env python
# coding: utf-8

# # Random Forest Classifier Multiclass

import sys
import time
import pandas as pd
#sys.path.append("..")
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

train_dataset = "./train_pp3_multi.csv"
test_dataset =  "./test_pp3_multi.csv"
def DF_XY_MULTI():
    """Loads preprocessed dataset files from pre-defined path, and splits into inputs and output.

    Returns:
        x_train, x_test, y_train, y_test: preprocessed splitted dataset
    """
    try:
        print("( 1 ) Reading Preprocessed CSV files..")
        print(f"Train Dataset: {train_dataset} is used!")
        print(f"Test Dataset: {test_dataset} is used!")
        train_multi = pd.read_csv(train_dataset)  # selected in code started line 17 if els
        print("\t Training dataset loaded..")
        test_multi = pd.read_csv(test_dataset)  # selected in code started line 17 if els
        print("\t Testing dataset loaded..\n")

        print("( 2 ) Loading done, splitting into X and Y..")
        x_train_multi, y_train_multi = train_multi.drop(["attack_cat"], axis=1), train_multi["attack_cat"]
        x_test_multi, y_test_multi = test_multi.drop(["attack_cat"], axis=1), test_multi["attack_cat"]
        print('\t ( 2.1 ) x_train Shape: ', '\t', x_train_multi.shape)
        print('\t ( 2.2 ) y_train Shape: ', '\t', y_train_multi.shape)
        print('\t ( 2.3 ) x_test Shape: ', '\t', x_test_multi.shape)
        print('\t ( 2.4 ) y_test Shape: ', '\t', y_test_multi.shape)

        print("( 3 ) Done!")
        print("PS! Import with: x_train_multi, x_test_multi, y_train_multi, y_test_multi = DF_XY_MULTI")

    except:
        print("Could not load dataset, try again..")
    return x_train_multi, x_test_multi, y_train_multi, y_test_multi


X_train_multi, X_test_multi, y_train_multi, y_test_multi = DF_XY_MULTI()

start_time = time.time()

# test min_samples_split: 1.0
params = {"criterion": "entropy",
              "bootstrap":True,
              "n_estimators": 200,
              "max_depth":14,
              "min_samples_split":2,
              "min_samples_leaf": 1,
              "n_jobs": -1}

# define the model
model = RandomForestClassifier(**params)
model.set_params(**params)

# fit the model on the whole dataset
model.fit(X_train_multi, y_train_multi)

# performing predictions on the traing and test dataset
y_pred_train = model.predict(X_train_multi)
y_pred_test = model.predict(X_test_multi)

# the formatting of the results
train_accuracy = accuracy_score(y_train_multi, y_pred_train)
test_accuracy = accuracy_score(y_test_multi, y_pred_test)
f1 = round(metrics.f1_score(y_test_multi, y_pred_test, average='weighted'), 5)
print(f"Training accuracy: \t{train_accuracy}\nTest accuracy: \t\t{test_accuracy}\nF1-score: \t\t{f1}\n")
print("------------------------------------------------------------------------------------------------")

train_accuracy = round(metrics.accuracy_score(y_train_multi, y_pred_train), 5)
test_accuracy = round(metrics.accuracy_score(y_test_multi, y_pred_test), 5)
weighted_f1 = round(metrics.f1_score(y_test_multi, y_pred_test, average='weighted'), 5)
weighted_precision = round(metrics.precision_score(y_test_multi, y_pred_test, average='weighted'), 5)
weighted_recall = round(metrics.recall_score(y_test_multi, y_pred_test, average='weighted'), 5)

elapsed_time = round((time.time() - start_time), 3)



# printing the output to the file RFC_results.txt
with open("RFC_results.txt", "a") as f:
    print(f"Training accuracy: \t{train_accuracy}\nTest accuracy: \t\t{test_accuracy}", file=f)
    print(f"Weighted F1-score: \t{weighted_f1}", file=f)
    print(f"Weighted precision: \t{weighted_precision}", file=f)
    print(f"Weighted recall: \t{weighted_recall}", file=f)
    print(f"Runtime: {elapsed_time}s",file=f)

#output = f"\t{train_accuracy}\n \t{test_accuracy}\n \t{weighted_f1}\n \t{weighted_precision}\n \t{weighted_recall}\n"

#output.to_csv("results.csv")




