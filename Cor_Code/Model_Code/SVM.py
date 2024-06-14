    #!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC

from sklearn import metrics

import sys
sys.path.append("..")

from sklearn.preprocessing import MinMaxScaler

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

"""
42 multi


Code is finished..
      C  acc_train  acc_test  prec_train  prec_test  F1_train   F1_test  rec_train  rec_test      Runtime
0  1.12   0.778226  0.699679    0.817939   0.820727  0.751492  0.706174   0.778226  0.699679  1161.447037




"""

"""
22 Multi
Code is finished..
      C  acc_train  acc_test  prec_train  prec_test  F1_train   F1_test  rec_train  rec_test     Runtime
0  1.12   0.866754  0.773476    0.847475   0.763394  0.829093  0.744105   0.866754  0.773476  286.879873

"""

"""
42 binary

Code is finished..
      C  acc_train  acc_test  prec_train  prec_test  F1_train   F1_test  rec_train  rec_test     Runtime
0  1.12   0.936096  0.814969     0.94041   0.857906  0.934177  0.804854   0.936096  0.814969  462.225241

"""

"""
21 Binary

      C  acc_train  acc_test  prec_train  prec_test  F1_train  F1_test  rec_train  rec_test     Runtime
0  1.12   0.935799  0.814288    0.940179   0.857754  0.933856  0.80403   0.935799  0.814288  510.279651

"""


# In[2]:

c = 1.12


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
   task2=["17FE", "41FE"] # 18 43 
   task3=["17FE", "41FE"] # ?17? 19 43
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
        
        
        Path = f'{Dataset}_SVM_{c}_{TYPE}_{FEATURES}.csv'
        
        if TYPE == "Multi":
           av = "weighted"
        else:
           av = "binary"
           
        x_train = pd.read_csv(f'../Dataset/Used/{Dataset}_xtrain_{FEATURES}_{TYPE}.csv')
        y_train = pd.read_csv(f'../Dataset/Used/{Dataset}_ytrain_{FEATURES}_{TYPE}.csv')
       
        x_test = pd.read_csv(f'../Dataset/Used/{Dataset}_xtest_{FEATURES}_{TYPE}.csv')
        y_test = pd.read_csv(f'../Dataset/Used/{Dataset}_ytest_{FEATURES}_{TYPE}.csv')
        
        X_train = scaler1.fit_transform(x_train)
        X_test = scaler2.fit_transform(x_test)
        
        # In[4]:
        
        y_test = np.ravel(y_test)
        y_train = np.ravel(y_train)
        
        error = []
        C_value = []
        
        acc_tr = []
        acc_te = []
        
        pre_tr = []
        pre_te = []
        
        F1_tr = []
        F1_te = []
        
        rec_tr = []
        rec_te = []
        
        timer = []
        
        import time
        
        def SVM_predict(c, svm_kernel, svm_gamma, svm_degree):
        
          print("Starting SVM....")
          SVM_model = SVC(kernel=svm_kernel, C = c, gamma=svm_gamma, degree=svm_degree)
          start = time.time()
          SVM_model.fit(X_train, y_train)
          
          pred_i = SVM_model.predict(X_train)
          pred_y = SVM_model.predict(X_test)
          end = time.time()
          # Appending values to list
          #error.append(np.mean(pred_i != y_test))
          C_value.append(c)
          acc_tr.append(metrics.accuracy_score(y_train, pred_i))
          acc_te.append(metrics.accuracy_score(y_test, pred_y))
            
          pre_tr.append(metrics.precision_score(y_train, pred_i, average = "weighted", zero_division=0.0))
          pre_te.append(metrics.precision_score(y_test, pred_y, average = "weighted", zero_division=0.0))
            
          F1_tr.append(metrics.f1_score(y_train, pred_i, average = "weighted", zero_division=0.0))
          F1_te.append(metrics.f1_score(y_test, pred_y, average="weighted", zero_division=0.0))
            
          rec_tr.append(metrics.recall_score(y_train, pred_i, average = "weighted", zero_division=0.0))
          rec_te.append(metrics.recall_score(y_test, pred_y, average="weighted", zero_division=0.0))
          timer.append(end-start)
          return pred_y, pred_i
        
        # In[8]:
        
        # Calling the function created above.
        model = SVM_predict(c=c, svm_kernel="rbf", svm_gamma="scale", svm_degree=3)
     
        # In[9]:
        
        print("SVM has finished!")
        print("Creating CSV file....")
        # Creating a dataframe and saving to file
        # dictionary of lists 
        dict = {
                'C': C_value, 
                'acc_train': acc_tr,
        	'acc_test': acc_te, 
                'prec_train': pre_tr,
        	'prec_test': pre_te,
                'F1_train': F1_tr,
        	'F1_test': F1_te,
                'rec_train': rec_tr,
        	'rec_test': rec_te,
        	'Runtime': timer,
                #'error': error
                }
        
        df = pd.DataFrame(dict, index=C_value)
        df.set_index("C", inplace = True)
        
        
        # In[]
        
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(y_test, model[0])
        print(cm)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        
        plt.show()
        
        # In[10]:
        
        # EXPORT AS CSV when done.
        df.to_csv(Path)
        
        print("Code is finished..")
        
        
        new_dataframe = pd.read_csv(Path)
        gg = new_dataframe.sort_values(by=["acc_test"], ascending=False)
        print(gg.to_string())
