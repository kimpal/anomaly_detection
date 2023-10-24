#!/usr/bin/env python
# coding: utf-8
# # Pre-processing (Complete)
# using label encoding on attack_cat
#%%
# ## imports
import os
import time
import warnings
import numpy as np  
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime


# 1. Reading Train and test dataset.
# 2. Check if the dataset is reversed.
#%%
train = pd.read_csv('../Dataset/UNSW_NB15_training-set.csv')
test = pd.read_csv('../Dataset/UNSW_NB15_testing-set.csv')
list_events = pd.read_csv('../Dataset/UNSW-NB15_LIST_EVENTS.csv')
features = pd.read_csv('../Dataset/NUSW-NB15_features.csv', encoding='cp1252')

#%%
print(train.shape, test.shape)
if train.shape[0]<100000:
    print("Train test sets are reversed. Fixing them.")
    train, test = test, train
    
# %%
train['type'] = 'train'
test['type'] ='test'
total = pd.concat([train, test], axis=0, ignore_index=True)
total.drop(['id'], axis=1, inplace=True)
# del train, test
# %%
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('object')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# %%
def standardize(df):
    return (df-df.mean())/df.std()
    
def min_max(df):
    return (df-df.min())/(df.max() - df.min())

def normalize(df):
    return pd.Dataframe(preprocessing.normalize(df), columns=df.columns)

# %%
total = reduce_mem_usage(total)
# %%
list_events.shape
# %%
list_events.head()
# %%
list_events['Attack category'].unique()
# %%
list_events['Attack subcategory'].unique()
# %%
features.head(features.shape[0])
# %%
# the Name column has camel case values
features['Name'] = features['Name'].str.lower()
# the following 4 columns are address related and not in train dataset
features = features[~features['Name'].isin(['srcip', 'sport', 'dstip', 'dsport'])].reset_index()
features.drop(['index', 'No.'], axis=1, inplace=True)
# %%
normal = train[train['label']==0]
anomaly = train[train['label']==1]

# %%
print(sorted(set(train.columns) - set(features['Name'].values)))
print(sorted(set(features['Name'].values) - set(train.columns)))
# %%
fix = {'ct_src_ ltm': 'ct_src_ltm', 'dintpkt': 'dinpkt', 'dmeansz': 'dmean', 'res_bdy_len': 'response_body_len', 'sintpkt': 'sinpkt', 'smeansz': 'smean'}
features['Name'] = features['Name'].apply(lambda x: fix[x] if x in fix else x)
features.to_csv('features.csv')
# %%
print(sorted(set(train.columns) - set(features['Name'].values)))
print(sorted(set(features['Name'].values) - set(train.columns)))
# %%
train.head()
# %%
train.dtypes
# %%
def show_correlation(data, method='pearson'):
    correlation_matrix = data.corr(method='pearson') #  ‘pearson’, ‘kendall’, ‘spearman’
    fig = plt.figure(figsize=(12,9))
    sns.heatmap(correlation_matrix,vmax=0.8,square = True) #  annot=True, if fig should show the correlation score too
    plt.show()
    return correlation_matrix

def top_correlations(correlations, limit=0.9):
    columns = correlations.columns
    for i in range(correlations.shape[0]):
        for j in range(i+1, correlations.shape[0]):
            if correlations.iloc[i,j] >= limit:
                print(f"{columns[i]} {columns[j]} {correlations.iloc[i,j]}")
def print_correlations(correlations, col1=None, col2=None):
    columns = correlations.columns
    for i in range(correlations.shape[0]):
        for j in range(i+1, correlations.shape[0]):
            if (col1 == None or col1==columns[i]) and (col2 == None or col2==columns[j]):
                print(f"{columns[i]} {columns[j]} {correlations.iloc[i,j]}")
                return
            elif (col1 == None or col1==columns[j]) and (col2 == None or col2==columns[i]):
                print(f"{columns[i]} {columns[j]} {correlations.iloc[i,j]}")
                return
            
def find_corr(df1, df2):
    return pd.concat([df1, df2], axis=1).corr().iloc[0,1]

def corr(col1, col2='label', df=total):
    return pd.concat([df[col1], df[col2]], axis=1).corr().iloc[0,1]
# %%
correlation_matrix = show_correlation(total)
# %%
top_correlations(correlation_matrix, limit=0.9)

# %%
correlation_matrix = show_correlation(train, method='spearman')
# %%
top_correlations(correlation_matrix, limit=0.9)
# %%
