# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:07:57 2023

@author: eriks
"""

import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# metrics are used to find accuracy or error
from sklearn import metrics


# In[2]:
    
def import_train_test_Mult():
    train = pd.read_csv('../Dataset/UNSW_NB15_training-set.csv')
    test = pd.read_csv('../Dataset/UNSW_NB15_testing-set.csv')
    
   
    
    # Dropping the columns based on Feature Selection:
    # https://www.kaggle.com/khairulislam/unsw-nb15-feature-importance
    drop_cols = ['id','label','response_body_len', 'spkts', 'ct_flw_http_mthd', 'trans_depth', 'dwin', 'ct_ftp_cmd', 'is_ftp_login']
    for df in [train, test]:
        for col in drop_cols:
            if col in df.columns:
                print('Dropping: ', col)
                df.drop([col], axis=1, inplace=True)
    
    
    static_train = pd.get_dummies(train['attack_cat'], prefix_sep='_', prefix='attack', drop_first=True, dtype= float)
    static_test = pd.get_dummies(test['attack_cat'], prefix_sep='_', prefix='attack', drop_first=True, dtype= float)
    
    train = pd.concat([train,static_train],axis=1)
    train.drop('attack_cat', axis=1, inplace=True)
    
    test = pd.concat([test,static_test],axis=1)
    test.drop('attack_cat', axis=1, inplace=True)
    
    
    
    if train.shape < test.shape:
        # Reversing the dataset
        train, test = test, train
        print("Train and Test sets are reversed, Corrected Shape:")
        print("Train shape: ", train.shape)
        print("Test shape: ", test.shape)
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
train, test = import_train_test_Mult()


# In[6]:


# To check if train and test datasets inhibits missing values
train.isnull().sum()
test.isnull().sum()


# In[7]:


# Addressing the different Data types for each column
train.dtypes
test.dtypes


# In[8]:

train.info()

train

test

# Splitting the dataset into inputs and outputs
y_train = train[['attack_Backdoor', 'attack_DoS', 'attack_Exploits', 'attack_Fuzzers', 'attack_Generic', 'attack_Normal', 'attack_Reconnaissance', 'attack_Shellcode', 'attack_Worms']]
y_test = test[['attack_Backdoor', 'attack_DoS', 'attack_Exploits', 'attack_Fuzzers', 'attack_Generic', 'attack_Normal', 'attack_Reconnaissance', 'attack_Shellcode', 'attack_Worms']]

x_train = train.drop(['attack_Backdoor', 'attack_DoS', 'attack_Exploits', 'attack_Fuzzers', 'attack_Generic', 'attack_Normal', 'attack_Reconnaissance', 'attack_Shellcode', 'attack_Worms'], axis=1)
x_test = test.drop(['attack_Backdoor', 'attack_DoS', 'attack_Exploits', 'attack_Fuzzers', 'attack_Generic', 'attack_Normal', 'attack_Reconnaissance', 'attack_Shellcode', 'attack_Worms'], axis=1)
# Running the inputs into the feature_engineer function
x_train, x_test = feature_engineer(x_train), feature_engineer(x_test)


# In[9]:

y_test
y_train

x_train

# Getting the categorical and non categorical columns
categorical_columns = get_cat_columns(x_train)
non_categorical_columns = [x for x in x_train.columns if x not in categorical_columns]


# In[10]:
categorical_columns

x_train.head()

y_test

# In[11]:


# Using standard scaler to normalize data on non categorical columns
scaler = StandardScaler()
x_train[non_categorical_columns] = scaler.fit_transform(x_train[non_categorical_columns])
x_test[non_categorical_columns] = scaler.transform(x_test[non_categorical_columns])


# In[12]:
y_test

y_train

x_train

print(pd.get_dummies(x_train , dtype= int))
# In[13]:


# Using get_dummies to make the categorical values usable.
x_train = pd.get_dummies(x_train , dtype= int)

x_train

x_test = pd.get_dummies(x_test, dtype= int)

print("Column mismatch {0}, {1}".format(set(x_train.columns)- set(x_test.columns),  set(x_test.columns)- set(x_train.columns)))
features = list(set(x_train.columns) & set(x_test.columns))

x_train

# In[14]:


features = list(set(x_train.columns) & set(x_test.columns))


# In[15]:


print(f"Number of features {len(features)}")
x_train = x_train[features]
x_test = x_test[features]


# In[16]:


x_train


# In[17]:


print('X_train Shape: ', x_train.shape)
print('y_train Shape: ', y_train.shape)
print('X_test Shape: ', x_test.shape)
print('y_test Shape: ', y_test.shape)

# ## Export CSV

# In[18]:


# merge x_train and y_train before exporting to CSV

x_train['attack_Backdoor'] = y_train['attack_Backdoor']
x_train['attack_DoS'] = y_train['attack_DoS']
x_train['attack_Exploits'] = y_train['attack_Exploits']

x_train['attack_Fuzzers'] = y_train['attack_Fuzzers']
x_train['attack_Generic'] = y_train['attack_Generic']
x_train['attack_Normal'] = y_train['attack_Normal']

x_train['attack_Reconnaissance'] = y_train['attack_Reconnaissance']
x_train['attack_Shellcode'] = y_train['attack_Shellcode']
x_train['attack_Worms'] = y_train['attack_Worms']



x_test['attack_Backdoor'] = y_test['attack_Backdoor']
x_test['attack_DoS'] = y_test['attack_DoS']
x_test['attack_Exploits'] = y_test['attack_Exploits']

x_test['attack_Fuzzers'] = y_test['attack_Fuzzers']
x_test['attack_Generic'] = y_test['attack_Generic']
x_test['attack_Normal'] = y_test['attack_Normal']

x_test['attack_Reconnaissance'] = y_test['attack_Reconnaissance']
x_test['attack_Shellcode'] = y_test['attack_Shellcode']
x_test['attack_Worms'] = y_test['attack_Worms']



x_train.to_csv('../Dataset/train_ppMULT_on_hot.csv', index=False)
x_test.to_csv('../Dataset/test_ppMULT_on_hot.csv', index=False)


# In[19]:
y_test

