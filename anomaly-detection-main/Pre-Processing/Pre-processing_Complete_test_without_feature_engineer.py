#!/usr/bin/env python
# coding: utf-8

# # Pre-processing (Complete)

# ## imports
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# metrics are used to find accuracy or error
from sklearn import metrics

# preprocessing and no reature enginering output 42 features by using lable encoding on all object types
# Functions
# 1. Reading Train and test dataset.
# 2. Check if dataset is reversed.
# 3. Drop 'id', and 'attack_cat' columns.
def import_train_test():
    #train = pd.read_csv('../../Anomaly-Detection-main/Dataset/UNSW_NB15_training-set.csv')
    #test = pd.read_csv('../../Anomaly-Detection-main//Dataset/UNSW_NB15_testing-set.csv')
    #print('dataset in shape of train: ', train.shape)
    #print('dataset in shape of tes: ', test.shape)
    #print(train.dtypes)
    #nan_count = train.isna().sum()
    #print(nan_count)
    train = pd.read_csv('../Dataset/train_data_pp.csv',low_memory=False)
    test = pd.read_csv('../Dataset/test_data_pp.csv', low_memory=False)
    print('dataset in shape of train: ', train.shape)
    print('dataset in shape of tes: ', test.shape)
    # Dropping the columns based on Feature Selection:
    # https://www.kaggle.com/khairulislam/unsw-nb15-feature-importance
    drop_cols = ['attack_cat', 'id']
    for df in [train, test]:
        # creating instance of labelencoder
        labelencoder = LabelEncoder()
        # Assigning numerical values to the column whit data type object and storing it in the same column
        #df['attack_cat'] = labelencoder.fit_transform(df['attack_cat'])
        df['proto'] = labelencoder.fit_transform(df['proto'])
        df['service'] = labelencoder.fit_transform(df['service'])
        df['state'] = labelencoder.fit_transform(df['state'])
        df['sport'] = labelencoder.fit_transform(df['sport']) # only present in large dataset
        df['dstip'] = labelencoder.fit_transform(df['dstip']) # only present in large dataset
        df['dsport'] = labelencoder.fit_transform(df['dsport']) # only present in large dataset
        df['srcip'] = labelencoder.fit_transform(df['srcip']) # only present in large dataset
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
    else:
        print("The dataset, is already reversed")
        print("Train shape: ", train.shape)
        print("Test shape: ", test.shape)
    return train, test


"""
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
"""
def get_cat_columns(train):
    # Defining an empty list
    categorical = []
    # Iterating through the columns and checking for columns with datatyp "Object"
    for col in train.columns:
        if train[col].dtype == 'object':
            categorical.append(col) # appending "object" columns to categorical
    return categorical


# ## Pre-processing

# Importing train test by using the function
train, test = import_train_test()

print("line91: train shap: ",train.shape)

# To check if train and test datasets inhibits missing values
print(train.isnull().sum())
print(test.isnull().sum())


# Addressing the different Data types for each column
print(train.dtypes)
print(test.dtypes)


# Splitting the dataset into inputs and outputs
x_train, y_train = train.drop(['label'], axis=1), train['label']
x_test, y_test = test.drop(['label'], axis=1), test['label']
# Running the inputs into the feature_engineer function
x_train, x_test = x_train, x_test #feature_engineer(x_train), feature_engineer(x_test) #train, test

print("line109 train shap: ",train.shape)
print("linex110 x_train shap ",x_train.shape)
# Getting the categorical and non categorical columns
categorical_columns = get_cat_columns(x_train)
non_categorical_columns = [x for x in x_train.columns if x not in categorical_columns]


print(x_train.head())
print("linex117 x_train shap ",x_train.dtypes)


# Using standard scaler to normalize data on non categorical columns
scaler = StandardScaler()
x_train[non_categorical_columns] = scaler.fit_transform(x_train[non_categorical_columns])
x_test[non_categorical_columns] = scaler.transform(x_test[non_categorical_columns])


print("linex126 x_train datatype ",x_train.dtypes)
print(x_train.head())

# consider removing
# Using get_dummies to make the categorical values usable.
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)
print("Column mismatch {0}, {1}".format(set(x_train.columns)- set(x_test.columns),  set(x_test.columns)- set(x_train.columns)))
features = list(set(x_train.columns) & set(x_test.columns))


features = list(set(x_train.columns) & set(x_test.columns))


print(f"Number of features {len(features)}")
x_train = x_train[features]
x_test = x_test[features]


print("line 136: \n",x_train.head())

print('X_train Shape: ', x_train.shape)
print('y_train Shape: ', y_train.shape)
print('X_test Shape: ', x_test.shape)
print('y_test Shape: ', y_test.shape)


# ## Export CSV
'''
for x in x_train.columns:    
    if x_train[x].dtype == float:
        x_train[x] = x_train[x].astype(int)
    
for x in x_test.columns:    
    if x_test[x].dtype == float:
        x_test[x] = x_test[x].astype(int)    
'''

# merge x_train and y_train before exporting to CSV
x_train['label'] = y_train
x_test['label'] = y_test


x_train.to_csv('../Dataset/train_pp3.csv', index=False)
x_test.to_csv('../Dataset/test_pp3.csv', index=False)

print("creating csv is done")
