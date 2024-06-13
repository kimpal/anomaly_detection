    #!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import sys
sys.path.append("..")
from sklearn.preprocessing import MinMaxScaler as MS
from sklearn.ensemble import BaggingClassifier as BC

from sklearn.model_selection import cross_validate
import time


scaler1 = MS()
scaler2 = MS()
# In[2]:

depth = 9
task1=["Binary", "Multi"]

start = 0
start2 = 0

data = 3

end = len(task1) #-1
end2= len(task1) #-1

if data ==1:
   Dataset = "UNSW_NB15"
   task2=["21FE", "42FE"]
   task3=["22FE", "42FE"]
if data == 2:
   Dataset = "TON_Train_Test"
   task2=["17FE", "41FE"] # 18 43                17 41
   task3=["17FE", "41FE"] # 17 19 43             17 41 
if data == 3:
   Dataset = "IoT_Botnet"
   task2=["9FE", "13FE"]  #13 17   --10    ----- 9 -- 13
   task3=["12FE", "16FE"]  #12 17           ------9 -- 13

for x in range(start, end):
   for y in range(start2, end2):

        FEATURES = task2[y]  #UNSW_NB15: 42FE, 22FE, 21FE Binary
                           #Bot: 17FE, 12FE Multi, 13FE Binary
                           #Ton: 43FE, 19FE Multi, 18FE Binary              
        print(Dataset)
        print(task1[x])
        
        if x == 1:
            FEATURES = task3[y]
            print(task3[y])
        else:
            print(task2[y])
        TYPE = task1[x] #Multi, Binary

        if TYPE == "Multi":
           av = "weighted"
        else:
           av = "binary"
        
        if Dataset == "UNSW_NB15" and FEATURES == "22FE" and TYPE == "Multi": 
          x_train = pd.read_csv(f'../Dataset/Used/{Dataset}_xtrain_{FEATURES}_{TYPE}_v1.csv')
          y_train = pd.read_csv(f'../Dataset/Used/{Dataset}_ytrain_{FEATURES}_{TYPE}_v1.csv')
          
          x_test = pd.read_csv(f'../Dataset/Used/{Dataset}_xtest_{FEATURES}_{TYPE}_v3.csv')
          y_test = pd.read_csv(f'../Dataset/Used/{Dataset}_ytest_{FEATURES}_{TYPE}_v3.csv')
        else:
          x_train = pd.read_csv(f'../Dataset/Used/{Dataset}_xtrain_{FEATURES}_{TYPE}.csv')
          y_train = pd.read_csv(f'../Dataset/Used/{Dataset}_ytrain_{FEATURES}_{TYPE}.csv')
        
          x_test = pd.read_csv(f'../Dataset/Used/{Dataset}_xtest_{FEATURES}_{TYPE}.csv')
          y_test = pd.read_csv(f'../Dataset/Used/{Dataset}_ytest_{FEATURES}_{TYPE}.csv')
        
        X_train = scaler1.fit_transform(x_train)
        X_test = scaler1.transform(x_test)
        #X_test = scaler2.fit_transform(x_test)
        
        print("Train: ", X_train.shape)
        print("Test: ", X_test.shape)
        
        # In[6]:
        
        X_train.shape
        y_train.shape
        
        # In[9]:
        
        #scoring = ["accuracy", "f1_macro"]
            
        clf = DecisionTreeClassifier(max_depth=depth) #
        model = BC(estimator=clf , n_estimators=50)
        if data == 1:
            start = time.time()
            classifier = model.fit(X_train, np.ravel(y_train))
            y_pred_tr = classifier.predict(X_train)
            y_pred_te = classifier.predict(X_test)
            end = time.time()
        else: 
            start = time.time()
            clf.fit(X_train, np.ravel(y_train))
            y_pred_tr = clf.predict(X_train)
            y_pred_te = clf.predict(X_test)
            end = time.time()
        #scores = cross_validate(clf, X_train, np.ravel(y_train), scoring = scoring, return_train_score=True ,cv=5)
        #print(np.average(scores["test_f1_macro"]))
        
        accuracy_train = metrics.accuracy_score(y_train, y_pred_tr)
        accuracy_test = metrics.accuracy_score(y_test, y_pred_te)
        
        
        print("\nAccuracy train:", accuracy_train)
        print("Accuracy test:", accuracy_test)
        
        f1_train = metrics.f1_score(y_train, y_pred_tr, average= av, zero_division=0.0)
        f1_test = metrics.f1_score(y_test, y_pred_te, average= av, zero_division=0.0)
        
        print("\nF1 train:", f1_train)
        print("F1 test:", f1_test)
        
        prec_te = metrics.precision_score(y_test, y_pred_te, average = av, zero_division=0.0)
        prec_tr = metrics.precision_score(y_train, y_pred_tr, average= av, zero_division=0.0)
        
        rec_tr = metrics.recall_score(y_train, y_pred_tr, average= av, zero_division=0.0)
        rec_te = metrics.recall_score(y_test, y_pred_te, average= av, zero_division=0.0)
        
        print("\nPrecision train:", prec_tr)
        print("Precision test: ", prec_te)
        
        print("\nRecall train:", rec_tr)
        print("Recall test:", rec_te)
        
        print("Runtime: ", str(end-start), "\n")
        
        print("Classification report: ")
        print(metrics.classification_report(y_test, y_pred_te))
        #print("------------------------------------------------")

"""
    a.append(metrics.accuracy_score(y_train, y_pred_tr))
    b.append(metrics.accuracy_score(y_test, y_pred_te))
    c.append(metrics.f1_score(y_train, y_pred_tr, average="weighted", zero_division=0.0))
    d.append(metrics.f1_score(y_test, y_pred_te, average="weighted", zero_division=0.0))
    e.append(metrics.precision_score(y_test, y_pred_te, average="weighted", zero_division=0.0))
    f.append(metrics.recall_score(y_test, y_pred_te, average="weighted", zero_division=0.0 ))

acc_tr = np.mean(a)
acc_te = np.mean(b)

f_tr = np.mean(c)
f_te = np.mean(d)

pr = np.mean(e)
re = np.mean(f)
"""
# In[10]:


"""
print("Average Accuracy Score Train: ", acc_tr)
print("Average Accuracy Score Test: ", acc_te)

print("Average F1 Score train: \t", f_tr)
print("Average F1 Score test: \t\t", f_te)

print("Average Precision Score: \t", pr)
print("Average Recall Score: \t\t", re)
"""

"""
42 multi

Train:  (175341, 42)
Test:  (82332, 42)

Accuracy train: 0.7975658859023275
Accuracy test: 0.6817276393139969

F1 train: 0.778094334800399
F1 test: 0.7053702620515548

Precision train: 0.8244262355154754
Precision test:  0.8126250692807774

Recall train: 0.7975658859023275
Recall test: 0.6817276393139969
Runtime:  86.49175095558167 

"""

"""
22 multi

Train:  (101204, 22)
Test:  (78963, 22)

Accuracy train: 0.8946879569977472
Accuracy test: 0.762027785165204

F1 train: 0.8787723970582385
F1 test: 0.768789591688343

Precision train: 0.9026971910403562
Precision test:  0.8024244770172628

Recall train: 0.8946879569977472
Recall test: 0.762027785165204
Runtime:  38.64880657196045 
"""

"""
42 binary

Train:  (175341, 42)
Test:  (82332, 42)

Accuracy train: 0.947878704923549
Accuracy test: 0.8515401059126464

F1 train: 0.9622721737156067
F1 test: 0.8798119942182322

Precision train: 0.9483620030269989
Precision test:  0.7936913442262317

Recall train: 0.9765964756454194
Recall test: 0.9868966734315715
Runtime:  77.86319422721863 

"""

"""
21 binary

Train:  (175341, 21)
Test:  (82332, 21)

Accuracy train: 0.9464985371362089
Accuracy test: 0.8310498955448671

F1 train: 0.961492518933563
F1 test: 0.8649409662886438

Precision train: 0.9424095144599836
Precision test:  0.7724765868886576

Recall train: 0.9813643257556078
Recall test: 0.9825509573810994
Runtime:  38.5535523891449 
"""