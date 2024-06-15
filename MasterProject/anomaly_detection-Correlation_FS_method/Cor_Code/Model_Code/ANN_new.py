#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import pandas as pd
import time
import sys
sys.path.append("..")
# from Functions.UNSW_DF_Multi_onehot import DF_XY_Multi
# from Functions.UNSW_DF_Multi_label import DF_XY_Multi


# from numpy import loadtxt
# from keras_visualizer import visualizer
# from ann_visualizer.visualize import ann_viz

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

# My Model

# 22_cor: 77.26 69.73
# 22_cor_smoterus: 86.25 71.72

# 42: 78.86, 72.93

# Sklearn
# MLP 22: 83.9, 83.27
# MLP 42: 83.7, 71.1

# Paper 19F: 79.46, 77.51
# Paper 42F: 79.91, 75.62

task1=["Binary", "Multi"]

start = 1
start2 = 0

data = 2

end = len(task1) #-1
end2= len(task1) -1

if data ==1:
   Dataset = "UNSW_NB15"
   task2=["21FE", "42FE"]
   task3=["22FE", "42FE"]
if data == 2:
   Dataset = "TON_Train_Test"
   task2=["17FE", "41FE"] # 18 43 
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
            FEATURES = task2[y]
            print(task2[y])
        TYPE = task1[x] #Multi, Binary


        Version = "v2" #v1(smote,rus) , v2(clean), v3(smote) v4(smote_rus_test) v1: 77.79 pre, 86.68, 77.80    v2: 71.9 pre: 86.59, 71.95  v3: 77 pre: 87.07, 77.11
        
        if TYPE == "Multi":
           av = "weighted"
        else:
           av = "binary"
                                                                                                # v4 : attack 9 : 11k samples
        x_train = pd.read_csv(f'../Dataset/Used/{Dataset}_xtrain_{FEATURES}_{TYPE}.csv')
        y_train = pd.read_csv(f'../Dataset/Used/{Dataset}_ytrain_{FEATURES}_{TYPE}.csv')
       
        x_test = pd.read_csv(f'../Dataset/Used/{Dataset}_xtest_{FEATURES}_{TYPE}.csv')
        y_test = pd.read_csv(f'../Dataset/Used/{Dataset}_ytest_{FEATURES}_{TYPE}.csv') 
                                                                                            # v1 + v3:  86.69 75.49  #100 88.52 78.37  -- 150 epoch: 89.88 78.08
                                                                                            # v1 + v4: 75.08  #100 88.49 75.08 20 #150: 89.14 77.24
                                                                                            # v1 + cor: 86.25 71.72   #100 88.23 74.62
                                                                                            # v1 + v2: 69.81     #100 -- 88.37 75.52  
                                                                                            # v1 + v1: 76.26 
                                                                                            
        x_train = scaler1.fit_transform(x_train)                                            # v4 + cor: 69.40 61.82
        x_test = scaler2.fit_transform(x_test)                                              # v4 + v3: 83 77
                                                                                            # v4 + v2: ----
                                                                                      # cor + v2: 76.94 73.5 #200 -- 79.29 76.85                                                                                    # cor + v3: 48
        print("\nTrain shape: ", x_train.shape)
        print("Test shape: ", x_test.shape)
        
        print(np.unique(y_train))
        print(len(np.unique(y_train)))
                            # relu, : 82.3, 77.5
                            #v1 + v4: relu 150 200ep lr 0.02 - 200 runs: 90.18 75.21 -- 10 runs: 89.15 75.89 
        runs = 1            #v1 + v3: relu 150 200ep lr 0.02 - 200 runs: 89.95 78.37 -- 10 runs: 88.87 76.45
        run2 = 10
        
        lr = 0.02 # 0.002
        na = 150 #100 #15 #150?
            
        act1 = 'relu' #relu  #softmax
        epoch = 20

# In[3]:

# Empty lists' for attributes
        ann_runtime = []



        def ANN_predict(e_start,e_end, batch_size):
            """Predicts an ANN model with a predefined model.
            Args:
                epoch_start (int): Start value for Epoch
                epoch_end (int): End value for Epoch
                bactch_size (int): Keras batch size
            """
            for z in range(0, runs):
                for epochs in range(e_start, e_end+1):
                    # define the keras model
                     model = Sequential()
                     model.add(
                         Dense(na, input_dim=x_train.shape[1], activation=act1))
                     if TYPE == "Multi":
                        model.add(Dense(len(np.unique(y_train)), activation='softmax'))
                        model.compile(loss='sparse_categorical_crossentropy',
                         optimizer=tf.keras.optimizers.Adam(learning_rate = lr), metrics=['accuracy'])
                     if TYPE == "Binary":
                        model.add(Dense(1, activation="sigmoid"))
                        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                         optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01), metrics=['accuracy'])
                     # compile the keras model
                     # predict_x = model.predict(x_test)
                     # Start time for calculating the runtime for each epoch
                     print(f"\n## ---------- EPOCH {epoch} ----------- ##\n")
                     # fit the keras model on the dataset
                     model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                               shuffle=True, validation_split=0.25, verbose=2)
            #print("Train mean val: ", np.mean(ann_train_accuracy))
            #print("Test mean val: ", np.mean(ann_test_accuracy))
            return model
        
        # In[4]:
            
        print("ANN start")
        start_time = time.time()
        model = ANN_predict(e_start= epoch,e_end=epoch, batch_size=128)
        elapsed_time = round((time.time() - start_time), 3)
        ann_runtime.append(elapsed_time)
        print(f"Runtime for Epoch {epoch}:\t {elapsed_time}s")
        # In[5]:
        
       # print(model.evaluate(x_test, y_test, verbose=0)[1]*100)
        
        # In[6]:
     #   print("MLP start")
     #   clf = MLPClassifier(hidden_layer_sizes=(na,),early_stopping = True, validation_fraction = 0.25, random_state=1, max_iter=20, learning_rate_init = lr, learning_rate = 'adaptive').fit(x_train, np.ravel(y_train))
     #   print("\nMLP train: ", clf.score(x_train, np.ravel(y_train)))
     #   print("MLP test: ", clf.score(x_test, np.ravel(y_test)))
        # In[5]:
        # evaluate the keras model for training and testing data
        a_a = []
        b_a = []
        
        a_f = []
        b_f = []
        
        a_p = []
        b_p = []
        
        a_r = []
        b_r = []
        
        a_t = []
        
        for c in range(0,run2):
            print(f'Evaluation {c} of {run2}')
            start = time.time()
            _, train_acc = model.evaluate(x_train, y_train, verbose=0)
            _, test_acc  = model.evaluate(x_test, y_test, verbose=0)
            a_a.append(train_acc*100)
            b_a.append(test_acc*100)
            
            if TYPE == "Multi":
                pred_tr = np.round(model.predict(x_train, verbose=0))
                pred_te = np.round(model.predict(x_test, verbose=0))
                pred_tr = np.argmax(pred_tr, axis=1)
                pred_te = np.argmax(pred_te, axis=1)
            else:
                pred_tr = (model.predict(x_train) >0.5).astype("int32")
                pred_te = (model.predict(x_test) >0.5).astype("int32")
            end = time.time()
            pred_tr = np.reshape(pred_tr,[-1,1])
            pred_te = np.reshape(pred_te,[-1,1])
            
            pr_tr = precision_score(y_train, pred_tr, average=av, zero_division=0.0)
            pr_te = precision_score(y_test, pred_te, average=av, zero_division=0.0)
            
            re_tr = recall_score(y_train, pred_tr, average=av, zero_division=0.0)
            re_te = recall_score(y_test, pred_te, average=av, zero_division=0.0)
            
            f1_tr = f1_score(y_train, pred_tr, average=av, zero_division=0.0)
            f1_te = f1_score(y_test, pred_te, average=av, zero_division=0.0)
               
            a_p.append(pr_tr*100)
            b_p.append(pr_te*100)
            
            a_r.append(re_tr*100)
            b_r.append(re_te*100)
        
            a_f.append(f1_tr*100)
            b_f.append(f1_te*100)
            
            a_t.append(end-start)
            
        print('\nANN average training\t: %.2f' % np.mean(a_a))
        print('ANN average testing\t: %.2f' % np.mean(b_a))
        
        print('\nANN average precision\t: %.2f' % np.mean(a_p))
        print('ANN average precision\t: %.2f' % np.mean(b_p))
        
        print('\nANN average recall\t: %.2f' % np.mean(a_r))
        print('ANN average recall\t: %.2f' % np.mean(b_r))
        
        print('\nANN average F1-score\t: %.2f' % np.mean(a_f))
        print('ANN average F1-score\t: %.2f' % np.mean(b_f))
        
        print('ANN average runtime\t: %.2f' % np.mean(a_t))

        print("\nX train: ", x_train.shape)
        print("X test: ", x_test.shape)
        
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        if TYPE == "Multi":
            cm = confusion_matrix(y_test, np.argmax(model.predict(x_test), axis=1))
        else:
            cm = confusion_matrix(y_test, (model.predict(x_test) > 0.5).astype("int32"))
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        
        plt.show()
# In[ ]:
"""
    # dictionary of lists
thisdict = {'epochs': ann_epochs,
        'no_layers': ann_no_of_layers,
        'accuracy_train': ann_train_accuracy,
        'accuracy_test': ann_test_accuracy,
        'error_train': ann_train_error,
        'error_test': ann_test_error,
        'precision_train': ann_train_precision,
        'precision_test': ann_test_precision,
            'F1_train': ann_train_f1,
        'F1_test': ann_test_f1,
            'recall_train': ann_train_recall,
            'recall_test': ann_test_recall,
            'runtime(s)': ann_runtime
            }

df = pd.DataFrame(thisdict)
# EXPORT AS CSV when done.
df.to_csv(f'{Dataset}_ANN_{epoch}E_(15)_AF({act1})_{FEATURES}_{TYPE}.csv', index=False)
"""

"""

#15 0.02 1

Accuracy for training:	 0.8526046401328011
Error for training:	 0.147395
Accuracy for testing:	 0.702006907652069
Error for testing:	 0.297993
Precision for training:	 0.903796
Precision for testing:	 0.842900
Recall for training:	 0.852605
Recall for testing:	 0.702007
F1 for training:	 0.848629
F1 for testing:		 0.747364
Runtime for Epoch 20:	 24.936s
Train mean val:  0.8526046401328011
Test mean val:  0.702006907652069
[0 1 2 3 4 5 6 7 8 9]
78.7489116191864

MLP train:  0.8769317418283862
MLP test:  0.769842733552411

ANN average training	: 88.15
ANN average testing	: 78.75


#150 20 0.02 10

ANN average training	: 89.17
ANN average testing	: 78.70

ANN average precision	: 92.74
ANN average precision	: 85.39

ANN average recall	: 87.34
ANN average recall	: 74.15

ANN average F1-score	: 86.96
ANN average F1-score	: 77.10

Runtime for Epoch 20:	 14.771s
78.69693636894226

MLP train:  0.8912987628947473
MLP test:  0.7711672590704849

X train:  (101204, 22)
X test:  (119288, 22)

[[    0     9     6   657     0     3     2     0     0     0]
 [    0    32     5   516     5     0    14     0     9     2]
 [   18    78    95  3426    41    55   108    35    76   157]
 [   82   168   145  8549   251   209   348    25   270  1085]
 [    8    27    10  1366  1582     8  2750    76    81   154]
 [    0    34    18   322    49 18258    46    18    17   109]
 [ 1369     5    20   524  2110     4 30965   396   825   782]
 [    0    13     0   378    15     5   422  1633   617   413]
 [    0     0     0     5    20     0    59    30   227    37]
 [    0     0     0  1433  1391     0  1192    31   803 32150]]
"""


"""
42 multi

Accuracy for training:	 0.6963288677491288
Error for training:	 0.303671
Accuracy for testing:	 0.6333746295486566
Error for testing:	 0.366625
Precision for training:	 0.824085
Precision for testing:	 0.828951
Recall for training:	 0.696329
Recall for testing:	 0.633375
F1 for training:	 0.731844
F1 for testing:		 0.698014
Runtime for Epoch 20:	 31.332s
Train mean val:  0.6963288677491288
Test mean val:  0.6333746295486566

ANN average training	: 78.41
ANN average testing	: 69.84
X train:  (175341, 42)
X test:  (82332, 42)
"""

"""
22 multi

Accuracy for training:	 0.8291767123829098
Error for training:	 0.170823
Accuracy for testing:	 0.706837379532186
Error for testing:	 0.293163
Precision for training:	 0.879761
Precision for testing:	 0.844841
Recall for training:	 0.829177
Recall for testing:	 0.706837
F1 for training:	 0.827817
F1 for testing:		 0.754128
Runtime for Epoch 20:	 29.965s
Train mean val:  0.8291767123829098
Test mean val:  0.706837379532186

78.8850486278534

MLP train:  0.8683846488281095
MLP test:  0.7760976659954663

ANN average training	: 86.26
ANN average testing	: 78.89

X train:  (101204, 22)
X test:  (78963, 22)
"""

"""
42 binary

Accuracy for training:	 0.9405729407269264
Error for training:	 0.059427
Accuracy for testing:	 0.88360540251664
Error for testing:	 0.116395
Precision for training:	 0.961902
Precision for testing:	 0.838167
Recall for training:	 0.950327
Recall for testing:	 0.977301
F1 for training:	 0.956079
F1 for testing:		 0.902403
Runtime for Epoch 20:	 33.301s
Train mean val:  0.9405729407269264
Test mean val:  0.88360540251664

ANN average training	: 94.06
ANN average testing	: 88.36
X train:  (175341, 42)
X test:  (82332, 42)
"""

"""
21 binary

Accuracy for training:	 0.9375388528638482
Error for training:	 0.062461
Accuracy for testing:	 0.8199242092989361
Error for testing:	 0.180076
Precision for training:	 0.921678
Precision for testing:	 0.757522
Recall for training:	 0.992576
Recall for testing:	 0.989764
F1 for training:	 0.955814
F1 for testing:		 0.858209
Runtime for Epoch 20:	 42.84s
Train mean val:  0.9375388528638482
Test mean val:  0.8199242092989361

ANN average training	: 93.75
ANN average testing	: 81.99
X train:  (175341, 21)
X test:  (82332, 21)
"""