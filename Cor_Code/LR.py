#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
from sklearn import metrics
#import seaborn as sns
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

import sys
sys.path.append("..")


# In[2]:


solv = 'lbfgs' #newton-cg
c = 1.0
m_it = 1000
data = 2

task1=["Binary", "Multi"]

start = 1
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
   task2=["17FE", "41FE"] # 17 18 43 
   task3=["17FE", "41FE"] # 17 19 43
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
          
        x_test.shape
        x_train.shape        
        #X_train = scaler1.fit_transform(x_train)
        #X_test = scaler2.fit_transform(x_test)
        # In[36]:
        
        # 300, 500, l2, lbfgs
        
        # multi_class='multinomial'
        logreg = LogisticRegression(max_iter = m_it, C = c, random_state=10, penalty = "l2", solver = solv)
        import time
        start = time.time()
        logreg.fit(x_train, np.ravel(y_train))
        y_pred_test = logreg.predict(x_test)
        y_pred_train = logreg.predict(x_train)
        end = time.time()
        
        print("\nAccuracy Score train: ", metrics.accuracy_score(y_train, y_pred_train))
        print("Accuracy Score test: ", metrics.accuracy_score(y_test, y_pred_test))
        
        print("\nF1 Score train: \t", metrics.f1_score(y_train, y_pred_train, average=av, zero_division= 0.0))
        print("F1 score test: \t\t", metrics.f1_score(y_test, y_pred_test, average=av, zero_division=0.0))
        
        print("\nPrecision score train: \t", metrics.precision_score(y_train, y_pred_train, average = av, zero_division=0.0))
        print("Precision Score test: \t", metrics.precision_score(y_test, y_pred_test, average = av, zero_division=0.0))
        
        print("\nRecall score train: \t", metrics.recall_score(y_train, y_pred_train, average = av, zero_division=0.0))
        print("Recall Score test: \t", metrics.recall_score(y_test, y_pred_test, average = av, zero_division= 0.0))
        
        print("\nRuntime: ", end-start)
        
        print("\n------------------------------------------------")
        print("Classification report: ")
        print(metrics.classification_report(y_test, y_pred_test, zero_division= 0.0))
        
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(y_test, y_pred_test)
        print(cm)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        
        plt.show()

#print(y_train.value_counts())
#print(y_test.value_counts())
"""
42 multi

Accuracy Score train:  0.7660501537005036
Accuracy Score test:  0.6748287421658651

F1 Score train: 	 0.7457853671881075
F1 score test: 		 0.6881615121130458

Precision score train: 	 0.7652688115006868
Precision Score test: 	 0.761668088437617

Recall score train: 	 0.7660501537005036
Recall Score test: 	 0.6748287421658651

Runtime:  66.80278205871582

------------------------------------------------
Classification report: 
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       677
           1       0.00      0.00      0.00       583
           2       0.17      0.04      0.06      4089
           3       0.52      0.77      0.62     11132
           4       0.25      0.64      0.36      6062
           5       0.98      0.94      0.96     18871
           6       0.95      0.61      0.74     37000
           7       0.39      0.73      0.51      3496
           8       0.00      0.00      0.00       378
           9       0.00      0.00      0.00        44

    accuracy                           0.67     82332
   macro avg       0.32      0.37      0.32     82332
weighted avg       0.76      0.67      0.69     82332

"""


"""
22 multi

Accuracy Score train:  0.8495019959685388
Accuracy Score test:  0.7409292959993922

F1 Score train: 	 0.830518033483334
F1 score test: 		 0.7370313509840244

Precision score train: 	 0.8279114987051794
Precision Score test: 	 0.7509331834069602

Recall score train: 	 0.8495019959685388
Recall Score test: 	 0.7409292959993922

Runtime:  16.866092681884766

------------------------------------------------
Classification report: 
              precision    recall  f1-score   support

           0       0.16      0.28      0.20       677
           1       0.05      0.01      0.01       583
           2       0.30      0.00      0.01      2044
           3       0.52      0.42      0.46      5566
           4       0.13      0.35      0.19      3031
           5       0.94      0.93      0.94      9436
           6       0.77      0.63      0.69     18500
           7       0.05      0.01      0.02      1748
           8       0.00      0.00      0.00       378
           9       0.87      0.93      0.90     37000

    accuracy                           0.74     78963
   macro avg       0.38      0.36      0.34     78963
weighted avg       0.75      0.74      0.74     78963

"""

"""
42 binary

Accuracy Score train:  0.9307634837259968
Accuracy Score test:  0.801753874556673

F1 Score train: 	 0.951178315772541
F1 score test: 		 0.8440891028580162

Precision score train: 	 0.914482790618548
Precision Score test: 	 0.7443729361816834

Recall score train: 	 0.990941922725635
Recall Score test: 	 0.9746536662843025

Runtime:  4.43594765663147

------------------------------------------------
Classification report: 
              precision    recall  f1-score   support

           0       0.95      0.59      0.73     37000
           1       0.74      0.97      0.84     45332

    accuracy                           0.80     82332
   macro avg       0.85      0.78      0.79     82332
weighted avg       0.84      0.80      0.79     82332

"""


"""
21 binary

Accuracy Score train:  0.9156272634466553
Accuracy Score test:  0.768498275275713

F1 Score train: 	 0.9405596091414613
F1 score test: 		 0.820395393980513

Precision score train: 	 0.9035099230395146
Precision Score test: 	 0.7160881723967758

Recall score train: 	 0.9807777712605056
Recall Score test: 	 0.9602708903203035

Runtime:  2.3097610473632812

------------------------------------------------
Classification report: 
              precision    recall  f1-score   support

           0       0.92      0.53      0.67     37000
           1       0.72      0.96      0.82     45332

    accuracy                           0.77     82332
   macro avg       0.82      0.75      0.75     82332
weighted avg       0.81      0.77      0.75     82332

"""