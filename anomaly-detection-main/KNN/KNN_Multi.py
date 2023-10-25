#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbor

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Dataset loading
import sys
sys.path.append("..")
from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()
scaler2 = StandardScaler()

# In[2]:


FEATURES = "21FE"   #42FE, 21FE, cor_42FE, cor_21FE

X_train = pd.read_csv(f'../Dataset/xtrain_{FEATURES}.csv')
y_train = pd.read_csv(f'../Dataset/ytrain_{FEATURES}.csv')

X_test = pd.read_csv(f'../Dataset/xtest_{FEATURES}.csv')
y_test = pd.read_csv(f'../Dataset/ytest_{FEATURES}.csv')

x_train = scaler1.fit_transform(X_train)
x_test = scaler2.fit_transform(X_test)

Max = 200

Path = f'K_SCORES(1-{Max})_Multi.csv'

# In[2]:


error_test = []
K_value = []
train_acc = []
test_acc = []
precision_score = []
F1_score = []
recall_score = []



x_train.shape
y_train.shape
# In[7]:
    

    
    
# Calculating error for K values between 1 and 200 and appending scores to lists

print(f"Running KNN (1-{Max})")

for i in range(1, Max):
    
    print("Round: ", i, " of ", Max)
    knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    knn.fit(x_train, np.ravel(y_train))
    
    pred_i_test = knn.predict(x_test)
    pred_i_train = knn.predict(x_train)
    
    # Appending values to list
    #error_test.append(np.mean(pred_i_test != y_test))
    K_value.append(i)
    train_acc.append(metrics.accuracy_score(y_train, pred_i_train))
    test_acc.append(metrics.accuracy_score(y_test, pred_i_test))
    precision_score.append(metrics.precision_score(y_test, pred_i_test, average="weighted", zero_division=0.0))
    F1_score.append(metrics.f1_score(y_test, pred_i_test, average="weighted", zero_division=0.0))
    recall_score.append(metrics.recall_score(y_test, pred_i_test, average="weighted", zero_division=0.0))
    
print("KNN run is over")

print("Creating a csv file....") 


# dictionary of lists 
dict = {
        'K': K_value,
        'train_acc': train_acc, 
        'test_acc': test_acc, 
        'precision': precision_score,
        'F1': F1_score,
        'recall': recall_score
        #'error': error_test
        }
# In[]
      
df = pd.DataFrame(dict, index=K_value)
df.set_index("K", inplace = True)
df
# EXPORT AS CSV when done.
df.to_csv(Path)    
print("Csv file has been created")    
    
# In[8]:

r'''
Max = 71

# Calculating error for K values between 1 and 200 and appending scores to lists

print("Running KNN (1-70)")

for i in range(1, Max):
    print("Round: ", i, " of ", Max)
    knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    knn.fit(x_train, y_train)
    
    pred_i_test = knn.predict(x_test)
    pred_i_train = knn.predict(x_train)
    
    # Appending values to list
    error_test.append(np.mean(pred_i_test != y_test))
    K_value.append(i)
    train_acc.append(metrics.accuracy_score(y_train, pred_i_train))
    test_acc.append(metrics.accuracy_score(y_test, pred_i_test))
    precision_score.append(metrics.precision_score(y_test, pred_i_test))
    F1_score.append(metrics.f1_score(y_test, pred_i_test))
    recall_score.append(metrics.recall_score(y_test, pred_i_test))
    
print("KNN run is over")

# In[4]:


# Creating a dataframe and saving to file

print("Creating a csv file....")

# dictionary of lists 
dict = {
        'K': K_value,
        'train_acc': train_acc, 
        'test_acc': test_acc, 
        'precision': precision_score,
        'F1': F1_score,
        'recall': recall_score,
        'error': error_test
        }

df = pd.DataFrame(dict, index=K_value)
df.set_index("K", inplace = True)
df
# EXPORT AS CSV when done.
df.to_csv('K_SCORES(1-70).csv')

print("Csv file has been created")


# In[5]:
    
plt.figure(figsize=(20, 12))
plt.plot(range(1, 71), error_test, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.savefig('KNN(1-70)_v3.png', dpi=40, transparent=True)
plt.show()    
# In[6]:

print("Opening csv file....")

dataframe = pd.read_csv(r"C:\Users\eriks\OneDrive - Østfold University College\Master\Master_Oppgave\Anomaly-Detection-main\KNN/K_SCORES(1-70).csv")    

dataframe
'''
# In[10]:


plt.figure(figsize=(20, 12))
plt.plot(range(1, Max), error_test, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.savefig(f'KNN(1-{Max})_v3.png', dpi=40, transparent=True)
plt.show()


# In[ ]:

print("Opening csv file....")

new_dataframe = pd.read_csv(Path)
new_dataframe.sort_values(by=['test_acc'], ascending = False)

