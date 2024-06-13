    #!/usr/bin/env python
# coding: utf-8

# # Random Forest Classifier Multiclass

import time
import pandas as pd
import numpy as np
import sys


sys.path.append("..")

from sklearn import metrics
# importing random forest classifier from assemble module

from sklearn.ensemble import RandomForestClassifier

# Tree Visualisation

from sklearn.preprocessing import MinMaxScaler as MS

scaler1 = MS()
scaler2 = MS()

# In[2]:


task1=["Binary", "Multi"]

start = 1
start2 = 0

data = 3

end = len(task1) 
end2= len(task1) -1

if data ==1:
   Dataset = "UNSW_NB15"
   task2=["21FE", "42FE"]
   task3=["22FE", "42FE"]
if data == 2:
   Dataset = "TON_Train_Test"
   task2=["17FE", "41FE"] # 17 18 43
   task3=["17FE", "41FE"] # 17 19 43
if data == 3:
   Dataset = "IoT_Botnet"
   task2=["9FE", "13FE"]  #13 17   --10    ----- 9 -- 13
   task3=["12FE", "16FE"]  #12 17           ------9 -- 13

for x in range(start, end):
   for y in range(start2, end2):

                           #UNSW_NB15: 42FE, 22FE, 21FE Binary
                           #Bot: 17FE, 12FE Multi, 13FE Binary
                           #Ton: 43FE, 19FE Multi, 18FE Binary
        RUNS = 1
        print(Dataset)
        print(task1[x])
        if x == 1:
            FEATURES = task3[y]
            print(task3[y])
        else:
            FEATURES = task2[y]
            print(task2[y])
            
        TYPE = task1[x] #Multi, Binary
        
       
        if TYPE == "Multi":
           av = "weighted"
        else:
           av = "binary"          
        
        if Dataset == "UNSW_NB15" and FEATURES == "22FE" and TYPE == "Multi": 
          x_train = pd.read_csv(f'../Dataset/Used/{Dataset}_xtrain_{FEATURES}_{TYPE}_cor.csv')
          y_train = pd.read_csv(f'../Dataset/Used/{Dataset}_ytrain_{FEATURES}_{TYPE}_cor.csv')
          
          x_test = pd.read_csv(f'../Dataset/Used/{Dataset}_xtest_{FEATURES}_{TYPE}_v2.csv')
          y_test = pd.read_csv(f'../Dataset/Used/{Dataset}_ytest_{FEATURES}_{TYPE}_v2.csv')
        else:
          x_train = pd.read_csv(f'../Dataset/Used/{Dataset}_xtrain_{FEATURES}_{TYPE}.csv')
          y_train = pd.read_csv(f'../Dataset/Used/{Dataset}_ytrain_{FEATURES}_{TYPE}.csv')
        
          x_test = pd.read_csv(f'../Dataset/Used/{Dataset}_xtest_{FEATURES}_{TYPE}.csv')
          y_test = pd.read_csv(f'../Dataset/Used/{Dataset}_ytest_{FEATURES}_{TYPE}.csv')
        
        X_train = scaler1.fit_transform(x_train)
        #X_test = scaler2.fit_transform(x_test)
        X_test = scaler1.fit_transform(x_test)
        # In[]
        
        y_train_multi = np.ravel(y_train)
        y_test_multi = np.ravel(y_test)
        X_test_multi = X_test
        X_train_multi = X_train
        
        train_accuracy = []
        
        test_accuracy = []
        
        f1_train = []
        f1_test = []
        
        prec_train = []
        prec_test = []
        
        rec_train = []
        rec_test = []
        
        elapsed_time = []
        
        print(X_train_multi.shape)
        print(X_test_multi.shape)
        
        def average(list, text):
         return print(f"Average {text} ", round((sum(list)/RUNS),3))
        
        
        
        for z in range(0, RUNS):
            # ## Creating the classifier
          print("----------------")
          print(f"Starting run {z+1} of {RUNS} runs.\n")
          start_time = time.time()
          model = RandomForestClassifier(criterion="entropy", bootstrap=True, n_estimators= 200, max_depth=50, min_samples_split=2, min_samples_leaf= 1, n_jobs= -1)
          # fit the model on the whole dataset
          model.fit(X_train_multi, y_train_multi)
          # performing predictions on the traing and test dataset
          y_pred_train = model.predict(X_train_multi)
          y_pred_test = model.predict(X_test_multi)
          elapsed_time.append(round((time.time() - start_time), 3))
        
          # Assuming y_train_multi, y_pred_train, y_test_multi, and y_pred_test are your multi-class targets and predictions
            
          train_accuracy.append(round(metrics.accuracy_score(y_train_multi, y_pred_train), 5))
          test_accuracy.append(round(metrics.accuracy_score(y_test_multi, y_pred_test), 5))
          f1_train.append(round(metrics.f1_score(y_train_multi, y_pred_train, average= av, zero_division=0.0),5))
          f1_test.append(round(metrics.f1_score(y_test_multi, y_pred_test, average=av, zero_division=0.0), 5))
          prec_train.append(round(metrics.precision_score(y_train_multi, y_pred_train, average=av, zero_division=0.0),5))
          prec_test.append(round(metrics.precision_score(y_test_multi, y_pred_test, average=av, zero_division=0.0), 5))
          rec_train.append(round(metrics.recall_score(y_train_multi, y_pred_train, average=av, zero_division=0.0), 5))
          rec_test.append(round(metrics.recall_score(y_test_multi, y_pred_test, average=av, zero_division=0.0), 5))
            
          #print(f"Training accuracy: \t{train_accuracy[x]}\nTest accuracy: \t\t{test_accuracy[x]}")
          #print(f"\nF1-score train: \t{f1_train[x]}\nF1-score test: \t\t{f1_test[x]}")
          #print(f"\nPrecision train: \t{prec_train[x]}\nPrecision test: \t{prec_test[x]}")
          #print(f"\nRecall train: \t\t{rec_train[x]}\nRecall test: \t\t{rec_test[x]}")
            
          
          #print(f"Runtime: {elapsed_time[x]}s")
            
        print("----------------------------------")
        
        average(train_accuracy, "train acc")
        average(test_accuracy, "test acc")
        
        average(f1_train, "f1 train acc")
        average(f1_test, "f1 test acc")
        
        average(prec_train, "precision train acc")
        average(prec_test, "precision test acc")
        
        average(rec_train, "recall train acc")
        average(rec_test, "recall test acc")
        
        average(elapsed_time, "runtime")
        
        print("----------------------------------")
        
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(y_test_multi, y_pred_test)
        print(cm)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        
        plt.show()