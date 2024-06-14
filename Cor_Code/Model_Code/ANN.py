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

Dataset = "UNSW_NB15" # IoT_Botnet, TON_Train_Test, UNSW_NB15

FEATURES = "22FE"  #UNSW_NB15: 42FE, 22FE, 21FE Binary
                   #Bot: 17FE, 12FE Multi, 13FE Binary 
                   #Ton: 43FE, 19FE Multi, 18FE Binary

TYPE = "Multi" #Multi, Binary

Version = "v3" #v1(smote,rus) , v2(clean), v3(smote) v4(smote_rus_test) v1: 77.79 pre, 86.68, 77.80    v2: 71.9 pre: 86.59, 71.95  v3: 77 pre: 87.07, 77.11

if TYPE == "Multi":
   av = "weighted"
else:
   av = "binary"
                                                                                        # v4 : attack 9 : 11k samples

x_train = pd.read_csv(f'../Dataset/Used/{Dataset}_xtrain_{FEATURES}_{TYPE}_v1.csv')    # cor 36 v1 # cor: 70.6 v2 # cor: 51 v3
y_train = pd.read_csv(f'../Dataset/Used/{Dataset}_ytrain_{FEATURES}_{TYPE}_v1.csv')    

x_test = pd.read_csv(f'../Dataset/Used/{Dataset}_xtest_{FEATURES}_{TYPE}_{Version}.csv')  
y_test = pd.read_csv(f'../Dataset/Used/{Dataset}_ytest_{FEATURES}_{TYPE}_{Version}.csv')   
                                                                                    # v1 + v3:  86.69 75.49  #100 88.52 78.37  -- 150 epoch: 89.88 78.08
                                                                                    # v1 + v4: 75.08  #100 88.49 75.08 20 #150: 89.14 77.24
                                                                                    # v1 + cor: 86.25 71.72   #100 88.23 74.62
                                                                                    # v1 + v2: 69.81     #100 -- 88.37 75.52  
                                                                                    # v1 + v1: 76.26 
                                                                                    
x_train = scaler1.fit_transform(x_train)                                            # v4 + cor: 69.40 61.82
x_test = scaler2.fit_transform(x_test)                                              # v4 + v3: 83 77
                                                                                    # v4 + v2: ----
                                                                                    # v4 + v1: ----
                                                                                    
                                                                                    # cor + cor: 61
                                                                                    # cor + v1: 35
                                                                                    # cor + v2: 76.94 73.5 #200 -- 79.29 76.85
                                                                                    # cor + v3: 48
print("Train: ", x_train.shape)
print("attacks: ", y_train.value_counts())
print("Test: ", x_test.shape)

np.unique(y_train)
np.unique(y_test)


# relu, : 82.3, 77.5
                            #v1 + v4: relu 150 200ep lr 0.02 - 200 runs: 90.18 75.21 -- 10 runs: 89.15 75.89 
runs = 1                    #v1 + v3: relu 150 200ep lr 0.02 - 200 runs: 89.95 78.37 -- 10 runs: 88.87 76.45
run2 = 10



lr = 0.02 # 0.002
na = 15 #100 #15 #150?

act1 = 'relu' #relu  #softmax
epoch = 20


train = pd.read_csv('../Dataset/train_pp4.csv')
test = pd.read_csv('../Dataset/test_pp4.csv')
     
     
     
x_train, y_train = train.drop(['attack_cat'], axis=1), train['attack_cat']
x_test, y_test = test.drop(['attack_cat'], axis=1), test['attack_cat']


y_test

print(x_train.shape)
print(y_train.value_counts())

print(x_test.shape)
print(y_test.value_counts())

# In[2]:


# ACCURACY
def get_accuracy_error(dataset, yhat_classes, train_test_string):
    """Get accuracy and error rate from given keras model.

    Args:
        dataset (dataframe): Dataframe for the model, either y_train or y_test
        yhat_classes (prediction): model prediction for crisp classes
        train_test_string (string): Represent string either as "train" or "test" for correct output

    Returns:
        float: returns accuracy and error rate for given model and datasets.
    """
    accuracy = accuracy_score(dataset, yhat_classes)

    if train_test_string == "train":
        # Train accuracy
        train_accuracy = accuracy
        print(f'Accuracy for training:\t {train_accuracy}')
        # Error rate for train
        train_error = 1 - train_accuracy
        print('Error for training:\t %f' % train_error)
        return train_accuracy, train_error
    elif train_test_string == "test":
        # Test accuracy
        test_accuracy = accuracy
        print(f'Accuracy for testing:\t {test_accuracy}')
        # Error rate for test
        test_error = 1 - test_accuracy
        print('Error for testing:\t %f' % test_error)
        return test_accuracy, test_error
    else:
        print("Type either 'train' or 'test' for train_test_string parameter")

# PRECISION

def get_precision(dataset, yhat_classes, train_test_string):
    """Get precision for given keras model: Precision = tp / (tp + fp)
    Args:
        dataset (dataframe): Dataframe for the model, either y_train or y_test
        yhat_classes (prediction): model prediction for crisp classes
        train_test_string (string): Represent string either as "train" or "test" for correct output

    Returns:
        float:  Returns precision for given model and dataset
    """
    precision = precision_score(
        dataset, yhat_classes, zero_division=0.0, average= av)

    if train_test_string == "train":
        train_precision = precision
        print('Precision for training:\t %f' % train_precision)
        return train_precision
    elif train_test_string == "test":
        test_precision = precision
        print('Precision for testing:\t %f' % test_precision)
        return test_precision
    else:
        print("Type either 'train' or 'test' for train_test_string parameter")

# RECALL

def get_recall(dataset, yhat_classes, train_test_string):
    """Get Recall for given keras model: Recall = tp / (tp + fn)
    Args:
        dataset (dataframe): Dataframe for the model, either y_train or y_test
        yhat_classes (prediction): model prediction for crisp classes
        train_test_string (string): Represent string either as "train" or "test" for correct output

    Returns:
        float:  Returns recall for given model and dataset
    """
    recall = recall_score(dataset, yhat_classes,
                           zero_division=0.0, average= av)

    if train_test_string == "train":
        train_recall = recall
        print('Recall for training:\t %f' % train_recall)
        return train_recall
    elif train_test_string == "test":
        test_recall = recall
        print('Recall for testing:\t %f' % test_recall)
        return test_recall
    else:
        print("Type either 'train' or 'test' for train_test_string parameter")

# F1


def get_F1(dataset, yhat_classes, train_test_string):
    """Get F1 for given keras model: F1 = 2 tp / (2 tp + fp + fn)
    Args:
        dataset (dataframe): Dataframe for the model, either y_train or y_test
        yhat_classes (prediction): model prediction for crisp classes
        train_test_string (string): Represent string either as "train" or "test" for correct output
    Returns:
        float:  Returns F1 for given model and dataset
    """
    f1 = f1_score(dataset, yhat_classes,zero_division=0.0, average= av)

    if train_test_string == "train":
        train_f1 = f1
        print('F1 for training:\t %f' % train_f1)
        return train_f1
    elif train_test_string == "test":
        test_f1 = f1
        print('F1 for testing:\t\t %f' % test_f1)
        return test_f1
    else:
        print("Type either 'train' or 'test' for train_test_string parameter")



from tensorflow.keras import backend as K

def f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# In[3]:

# Empty lists' for accuracy
ann_train_accuracy, ann_test_accuracy = [], []
# Empty lists' for precision
ann_train_precision, ann_test_precision = [], []
# Empty lists' for accuracy
ann_train_recall, ann_test_recall = [], []
# Empty lists' for F1
ann_train_f1, ann_test_f1 = [], []
# Empty lists' for error rate
ann_train_error, ann_test_error = [], []
# Empty lists' for attributes
ann_epochs, ann_no_of_layers, ann_runtime = [], [], []

metric = keras.metrics.F1Score(threshold=0.5)

def ANN_predict(e_start,e_end, batch_size):
    """Predicts an ANN model with a predefined model.
    Args:
        epoch_start (int): Start value for Epoch
        epoch_end (int): End value for Epoch
        bactch_size (int): Keras batch size
    """
    for x in range(0, runs):
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
             start_time = time.time()
             print(f"\n## ---------- EPOCH {epoch} ----------- ##\n")
             # fit the keras model on the dataset
             model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                       shuffle=True, validation_split=0.25, verbose=2)
                
             ### ------------------------###
             ### --- MODEL PREDICTION ---###
             ### ------------------------###
             if TYPE == "Multi":
                yhat_probs_train = np.round(model.predict(x_train, verbose=0))
                yhat_probs_test = np.round(model.predict(x_test, verbose=0))

                yhat_classes_train = np.argmax(yhat_probs_train, axis=1)
                yhat_classes_test = np.argmax(yhat_probs_test, axis=1)
             else:
                yhat_classes_train = (model.predict(x_train) >0.5).astype("int32")
                yhat_classes_test = (model.predict(x_test) >0.5).astype("int32")
                  #yhat_classes_train = (yhat_probs_train, axis=1)
                  #yhat_classes_test =  (yhat_probs_test, axis=1)
             ### --------------------###
             ### --- MODEL SCORES ---###
             ### --------------------###
             # Label
             used_train = yhat_classes_train
             used_test = yhat_classes_test
             ### --- ACCURACY & ERROR ---###
             train_accuracy, train_error = get_accuracy_error(
                 y_train, used_train, "train")
             test_accuracy, test_error = get_accuracy_error(
                 y_test, used_test, "test")
             ### --- PRECISION ---###
             train_precision = get_precision(y_train, used_train, "train")
             test_precision = get_precision(y_test, used_test, "test")
             ### --- RECALL ---###
             train_recall = get_recall(y_train, used_train, "train")
             test_recall = get_recall(y_test, used_test, "test")
             ### --- F1 ---###
             train_f1 = get_F1(y_train, used_train, "train")
             test_f1 = get_F1(y_test, used_test, "test")
             
             # Appending Scores to lists'
             ann_train_accuracy.append(train_accuracy)
             ann_test_accuracy.append(test_accuracy)

             ann_train_error.append(train_error)
             ann_test_error.append(test_error)

             ann_train_precision.append(train_precision)
             ann_test_precision.append(test_precision)

             ann_train_recall.append(train_recall)
             ann_test_recall.append(test_recall)

             ann_train_f1.append(train_f1)
             ann_test_f1.append(test_f1)

             # Appending attributes
             ann_no_of_layers.append(len(model.layers))
             ann_epochs.append(epoch)
            
             elapsed_time = round((time.time() - start_time), 3)
             ann_runtime.append(elapsed_time)
             print(f"Runtime for Epoch {epoch}:\t {elapsed_time}s")
    #print("Train mean val: ", np.mean(ann_train_accuracy))
    #print("Test mean val: ", np.mean(ann_test_accuracy))
    return model

# In[4]:
    
print("ANN start")
model = ANN_predict(e_start= epoch,e_end=epoch, batch_size=128)

# In[5]:
# # Neural Network

#y_t = model.predict(x_test)
#y_te = model.predict(x_train)

#print(np.unique(y_te))

#y_t = np.argmax(y_t, axis=1)
#y_te = np.argmax(y_te, axis=1)

#print(np.unique(y_te))

print(np.unique(y_train))

print(model.evaluate(x_test, y_test, verbose=0)[1]*100)

# In[6]:
print("MLP start")
clf = MLPClassifier(hidden_layer_sizes=(na,),early_stopping = True, validation_fraction = 0.25, random_state=1, max_iter=20, learning_rate_init = lr, learning_rate = 'adaptive').fit(x_train, np.ravel(y_train))
print("\nMLP train: ", clf.score(x_train, np.ravel(y_train)))
print("MLP test: ", clf.score(x_test, np.ravel(y_test)))
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

for x in range(0,run2):
    print(f'Evaluation {x} of {run2}')
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
    
print('\nANN average training\t: %.2f' % np.mean(a_a))
print('ANN average testing\t: %.2f' % np.mean(b_a))

print('\nANN average precision\t: %.2f' % np.mean(a_p))
print('ANN average precision\t: %.2f' % np.mean(b_p))

print('\nANN average recall\t: %.2f' % np.mean(a_r))
print('ANN average recall\t: %.2f' % np.mean(b_r))

print('\nANN average F1-score\t: %.2f' % np.mean(a_f))
print('ANN average F1-score\t: %.2f' % np.mean(b_f))

print("\nX train: ", x_train.shape)
print("X test: ", x_test.shape)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
if TYPE == "Multi":
    cm = confusion_matrix(y_test, np.argmax(model.predict(x_test), axis=1))
else:
    cm = confusion_matrix(y_test, (model.predict(x_test) > 0.5).astype("int32"), axis=1)
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