#!/usr/bin/env python
# coding: utf-8

# # Pre-processing (Complete)
#Binary TON_IoT

# imports
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
# metrics are used to find accuracy or error
from sklearn import metrics

labelencoder = LabelEncoder()
# Functions
# comment out the splitting the train_test since it is not nesesery to do it multiple times:
#train_test = pd.read_csv('../Dataset/Ton_IoT/Train_Test_Network.csv') #added to work whit TON_IoT
#train, test = train_test_split(train_test, test_size=0.40) #added to work whit TON_IoT
#train.to_csv('../Dataset/Ton_IoT/train.csv', index=False) #added to work whit TON_IoT
#test.to_csv('../Dataset/Ton_IoT/test.csv', index=False) #added to work whit TON_IoT
# 1. Reading Train and test dataset.
# 2. Check if dataset is reversed.
# 3. Drop 'id', and 'attack_cat' columns.
Bot_IoT_drop = ['Stime','saddr','daddr','subcategory', 'category']
TON_IoT_drop =['weird_notice','weird_addl','Weirdname','ts','src_ip','dst_ip','type']
#target= 'attack' # on Bot_IoT the target with the binary attack is named label
target= 'label' # on Ton_IoT the target with the binary attack is named label
def import_train_test():
    train = pd.read_csv('../Dataset/Ton_IoT/train.csv')
    test = pd.read_csv('../Dataset/Ton_IoT/test.csv')
    print("train: ",train.shape)
    print("test: ",test.shape)
    # Dropping the columns based on Feature Selection:
    # https://www.kaggle.com/khairulislam/unsw-nb15-feature-importance
    drop_cols = ['id'] +TON_IoT_drop #+ ['response_body_len', 'spkts', 'ct_flw_http_mthd', 'trans_depth', 'dwin', 'ct_ftp_cmd', 'is_ftp_login']
    for df in [train, test]:
        # creating instance of label encoder
        # Assigning numerical values and storing in the same column
        #df[target] = labelencoder.fit_transform(df[target])
        for col in drop_cols:
            if col in df.columns:
                print('Dropping: ', col)
                df.drop([col], axis=1, inplace=True)
    # no data flip
    #train, val = train_test_split(train, test_size=0.25)
    #return train, val, test

    # data flip
    #"""
    if train.shape < test.shape:
        # Reversing the dataset
        train, test = test, train
        print("Train and Test sets are reversed, Corrected Shape:")
        print("Train shape: ", train.shape)
        print("Test shape: ", test.shape)
        train, val = train_test_split(train, test_size=0.25)
    else:
        print("The dataset, is already reversed")
        print("Train shape: ", train.shape)
        print("Test shape: ", test.shape)
        train, val = train_test_split(train, test_size=0.25)
    return train, val, test
    #"""

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


# Importing train test by using the import_train_test function
train, val, test = import_train_test()

print("validation split is and column drop is done")
print("train ",train.shape)
print("validation ",val.shape)
print("test ",test.shape)

# In[6]:
# Use value_counts to get the counts of each unique value in train, val, test
def print_attack_distribution(dataset, dataset_name):
    attack_counts = dataset[target].value_counts()

    # Print the distribution
    print(f"\nAttack class distribution {dataset_name}:")
    for attack, count in attack_counts.items():
        percentage = count / len(dataset[target]) * 100
        print(f"Class={attack}: n={count} ({percentage:.3f}%)")

# Calling the print_attack_distribution on train, val and test,
print_attack_distribution(train, 'train')
print_attack_distribution(val, 'val')
print_attack_distribution(test, 'test')
print("\n")

# To check if train and test datasets inhibits missing values
train.isnull().sum()
val.isnull().sum()
test.isnull().sum()


# In[7]:


# Addressing the different Data types for each column
train.dtypes
val.dtypes
test.dtypes


# In[ ]:


# Splitting the dataset into inputs and outputs
x_train, y_train = train.drop([target], axis=1), train[target]
x_val, y_val = val.drop([target], axis=1), val[target]
x_test, y_test = test.drop([target], axis=1), test[target]
# Running the inputs into the feature_engineer function
#x_train, x_val, x_test = feature_engineer(x_train),feature_engineer(x_val), feature_engineer(x_test)



# Getting the categorical and non categorical columns
categorical_columns = get_cat_columns(x_train)
non_categorical_columns = [x for x in x_train.columns if x not in categorical_columns]


x_train.head()


# Using standard scaler to normalize data on non categorical columns
scaler = StandardScaler()
# Experimental scaling on Logistic regression
#scaler = MinMaxScaler()
x_train[non_categorical_columns] = scaler.fit_transform(x_train[non_categorical_columns])
x_val[non_categorical_columns] = scaler.transform(x_val[non_categorical_columns])
x_test[non_categorical_columns] = scaler.transform(x_test[non_categorical_columns])


x_train


# Using get_dummies to make the categorical values usable.
x_train = pd.get_dummies(x_train)
x_val = pd.get_dummies(x_val)
x_test = pd.get_dummies(x_test)
print("Column mismatch {0}, {1}".format(set(x_train.columns)- set(x_test.columns),  set(x_test.columns)- set(x_train.columns), set(x_train.columns)- set(x_val.columns),  set(x_val.columns)- set(x_train.columns)))
features = list(set(x_train.columns) & set(x_val.columns) & set(x_test.columns))


# In[ ]:


features = list(set(x_train.columns)& set(x_val.columns) & set(x_test.columns))


# In[ ]:


print(f"Number of features {len(features)}")
x_train = x_train[features]
x_val = x_val[features]
x_test = x_test[features]


# In[ ]:


x_train


# In[ ]:


print('X_train Shape: ', x_train.shape)
print('y_train Shape: ', y_train.shape)
print('X_val Shape: ', x_val.shape)
print('y_val Shape: ', y_val.shape)
print('X_test Shape: ', x_test.shape)
print('y_test Shape: ', y_test.shape)


# ## Export CSV
# In[ ]:
'''
for x in x_train.columns:    
    if x_train[x].dtype == float:
        x_train[x] = x_train[x].astype(int)
    
for x in x_test.columns:    
    if x_test[x].dtype == float:
        x_test[x] = x_test[x].astype(int)    
'''  
# In[ ]:


# merge x_train and y_train before exporting to CSV
x_train[target] = y_train
x_val[target] = y_val
x_test[target] = y_test


x_train.to_csv('../Dataset/Ton_IoT/train_pp_binary.csv', index=False)
x_val.to_csv('../Dataset/Ton_IoT/val_pp_binary.csv', index=False)
x_test.to_csv('../Dataset/Ton_IoT/test_pp_binary.csv', index=False)
print(x_train.dtypes)


print(x_train)
print("-----------------------------------------------")
print(x_val)
print(x_test)
print(x_train[target].unique())
print(x_train.dtypes)
print(x_val.dtypes)
print(x_test.dtypes)
# In[ ]:
y_test
# In[ ]:
y_test
