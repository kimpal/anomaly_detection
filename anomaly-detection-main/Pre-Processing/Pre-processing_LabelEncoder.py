#!/usr/bin/env python
# coding: utf-8

# # Pre-processing (Complete)

# ## imports

# In[1]:


import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# metrics are used to find accuracy or error
from sklearn import metrics

# creating instance of labelencoder
labelencoder = LabelEncoder()

# ## Functions

# In[2]:


# 1. Reading Train and test dataset.
# 2. Check if dataset is reversed.
# 3. Drop 'id', and 'attack_cat' columns.

saveTrain = '../Dataset/train_label_multi_10_classes.csv'
saveTest = '../Dataset/test_label_multi_10_classes.csv'

def import_train_test():
    train = pd.read_csv('../Dataset/UNSW_NB15_training-set.csv')
    test = pd.read_csv('../Dataset/UNSW_NB15_testing-set.csv')
    
    # Dropping the columns based on Feature Selection:
    # https://www.kaggle.com/khairulislam/unsw-nb15-feature-importance
    drop_cols = ['id','label'] 
    #+ ['response_body_len', 'spkts', 'ct_flw_http_mthd', 'trans_depth', 'dwin', 'ct_ftp_cmd', 'is_ftp_login']
    
    
    # backdoor, analysis, reconnaissance, shellcode, worms, and DOS -- https://ieeexplore.ieee.org/abstract/document/9751832
    # added Fuzzers too
    print(train['attack_cat'].value_counts())
    
    #train['attack_cat'] = train['attack_cat'].replace(['Worms','Shellcode','Reconnaissance','Analysis','Backdoor','DoS'],'WSRABD')
    #test['attack_cat'] = train['attack_cat'].replace(['Worms','Shellcode','Reconnaissance','Analysis','Backdoor','DoS'],'WSRABD')
    
    print(train['attack_cat'].value_counts())
    
    print(train.shape)
    
    for df in [train, test]:
        # Assigning numerical values and storing in same column
        df['attack_cat'] = labelencoder.fit_transform(df['attack_cat'])
        for col in drop_cols:
            if col in df.columns:
                print('Dropping: ', col)
                df.drop([col], axis=1, inplace=True)
    
    if train.shape < test.shape:
        # Reversing the dataset
        train, test = test, train
        print("Train and Test sets are reversed, Corrected Shape:")
        print("Train shape: ", train.shape)
        print("Test shape: ", test.shape)
        print("train attack category numeric", train.attack_cat.unique())
        print("test attack category numeric", test.attack_cat.unique())
    else:
        print("The dataset, is already reversed")
        print("Train shape: ", train.shape)
        print("Test shape: ", test.shape)
    return train, test


# In[3]:


def feature_engineer(df):
    # Everything except: 'FIN', 'INT', 'CON', 'REQ', 'RST is renamed 'others'
    df.loc[~df['state'].isin(['FIN', 'INT', 'CON', 'REQ', 'RST']), 'state'] = 'others'
    # Everything except: ''-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh', 'pop3' is renamed 'others'
    df.loc[~df['service'].isin(['-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh', 'pop3']), 'service'] = 'others'
    # Merging 'igmp', 'icmp', 'rtp' into one protocol: 'igmp_icmp_rtp'
    df.loc[df['proto'].isin(['igmp', 'icmp', 'rtp']), 'proto'] = 'igmp_icmp_rtp'
    # Everything except: 'tcp', 'udp' ,'arp', 'ospf', 'igmp_icmp_rtp' is renamed to 'others'
    df.loc[~df['proto'].isin(['tcp', 'udp','arp', 'ospf', 'igmp_icmp_rtp']), 'proto'] = 'others'
    return df


# In[4]:


def get_cat_columns(train):
    # Defining an empty list
    categorical = []
    # Iterating through the columns and checking for columns with datatyp "Object"
    for col in train.columns:
        if train[col].dtype == 'object':
            categorical.append(col) # appending "object" columns to categorical
    return categorical


# ## Pre-processing

# In[5]:


# Importing train test by using the function
train, test = import_train_test()

print(train['attack_cat'].value_counts())
# In[6]:


# To check if train and test datasets inhibits missing values
train.isnull().sum()
test.isnull().sum()


# In[7]:


# Addressing the different Data types for each column
train.dtypes
test.dtypes


# In[ ]:


# Splitting the dataset into inputs and outputs
x_train, y_train = train.drop(['attack_cat'], axis=1), train['attack_cat']
x_test, y_test = test.drop(['attack_cat'], axis=1), test['attack_cat']
# Running the inputs into the feature_engineer function
#x_train, x_test = feature_engineer(x_train), feature_engineer(x_test)


print(y_train.value_counts())

# In[ ]:


# Getting the categorical and non categorical columns
categorical_columns = get_cat_columns(x_train)

print(categorical_columns)

print(x_train[categorical_columns])

for categorical_columns in x_train:
    x_train[categorical_columns] = labelencoder.fit_transform(x_train[categorical_columns])
    x_test[categorical_columns] = labelencoder.fit_transform(x_test[categorical_columns])

non_categorical_columns = [x for x in x_train.columns if x not in categorical_columns]


# In[ ]:

print(x_test)
print(x_train)

# In[ ]:


# Using standard scaler to normalize data on non categorical columns
scaler = StandardScaler()
x_train[non_categorical_columns] = scaler.fit_transform(x_train[non_categorical_columns])
x_test[non_categorical_columns] = scaler.transform(x_test[non_categorical_columns])


# In[ ]:

print(y_train.value_counts())


# In[ ]:

"""
# Using get_dummies to make the categorical values usable.
x_train = pd.get_dummies(x_train, dtype=int)
x_test = pd.get_dummies(x_test, dtype=int)
print("Column mismatch {0}, {1}".format(set(x_train.columns)- set(x_test.columns),  set(x_test.columns)- set(x_train.columns)))
features = list(set(x_train.columns) & set(x_test.columns))
"""

# In[ ]:

"""
features = list(set(x_train.columns) & set(x_test.columns))


# In[ ]:


print(f"Number of features {len(features)}")
x_train = x_train[features]
x_test = x_test[features]
"""

# In[ ]:



# In[ ]:


print('X_train Shape: ', x_train.shape)
print('y_train Shape: ', y_train.shape)
print('X_test Shape: ', x_test.shape)
print('y_test Shape: ', y_test.shape)


# ## Export CSV
# In[ ]:
"""
for x in x_train.columns:    
    if x_train[x].dtype == float:
        x_train[x] = x_train[x].astype(int)
    
for x in x_test.columns:    
    if x_test[x].dtype == float:
       x_test[x] = x_test[x].astype(int)    
"""
# In[ ]:


# merge x_train and y_train before exporting to CSV
x_train['attack_cat'] = y_train
x_test['attack_cat'] = y_test


x_train.to_csv(saveTrain, index=False)
x_test.to_csv(saveTest, index=False)


x_train.info()

# In[ ]:
