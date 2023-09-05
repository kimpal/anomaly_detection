#!/usr/bin/env python
# coding: utf-8

# # Feature Importance

# ## (1) Importing libraries

# In[1]:


# Data manipulation and visualization
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn libraries.
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


# Gradient boosting framework(tree based learning algorithms).
#import lightgbm as lgb
# For creating progress meters.
from tqdm import tqdm_notebook as tqdm

# Misc imports
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)


# ## (2) Defining Functions

# In[2]:


def get_filenames(path):
    """Function to print out 
    all available files at given path.

    Args:
        path (str): print out all files in path
    """
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if 'csv' in filename:
                print(os.path.join(dirname, filename))
                
                
def get_train_test():
    """ (1) imports training and testing datasets.
        (2) Correcting reversed datasets.
        (3) Dropping "attack_Cat", and "id" columns.

    Returns:
       train, test: dataframes
    """
    root = "../Dataset/"
    train = pd.read_csv(root+'UNSW_NB15_training-set.csv')
    test = pd.read_csv(root+'UNSW_NB15_testing-set.csv')
    
    cols_to_drop = ['attack_cat', 'id']
    
    if train.shape < test.shape:
        print("Training and testing sets are reveresed. Correcting..")
        train, test = test, train
        print(f"✅ Corrected training shape:\t {train.shape}\n✅ Corrected testing shape:\t {test.shape}\n")

    for df in [train, test]:
        for col in cols_to_drop:
            if col in df.columns:
                print(f"❌ Dropped:\t {col}")
                df.drop([col], axis=1, inplace=True)
    return train, test

def get_categorical_columns(train):
    """inputs training set and returns a list of columns of dtype object.

    Args:
        train (dataframe): dataframe in

    Returns:
        list: returns a list with columns of dtype object.
    """
    categorical_columns = []
    for col in train.columns:
        if train[col].dtype == 'object':
            categorical_columns.append(col)
    return categorical_columns


def label_encode(train, test):
    """ Label encodes categorical columns in dataframes

    Args:
        train (dataframe): dataframe in
        test (dataframe): dataframe in

    Returns:
        train, test: label encoded dataframes
    """
    for col in get_categorical_columns(train):
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))
    return train, test

def feature_engineer(df):
    """ feature engineering for input labels. Combining several and columns.

    Args:
        df (dataframe): Dataframe in

    Returns:
        dataframe: returns a feature engineered dataframe.
    """
    # Everything except: 'FIN', 'INT', 'CON', 'REQ', 'RST is renamed 'others'
    df.loc[~df['state'].isin(['FIN', 'INT', 'CON', 'REQ', 'RST']), 'state'] = 'others'
    # Everything except: ''-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh', 'pop3' is renamed 'others'
    df.loc[~df['service'].isin(['-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh', 'pop3']), 'service'] = 'others'
    # Merging 'igmp', 'icmp', 'rtp' into one protocol: 'igmp_icmp_rtp'
    df.loc[df['proto'].isin(['igmp', 'icmp', 'rtp']), 'proto'] = 'igmp_icmp_rtp'
    # Everything except: 'tcp', 'udp' ,'arp', 'ospf', 'igmp_icmp_rtp' is renamed to 'others'
    df.loc[~df['proto'].isin(['tcp', 'udp','arp', 'ospf', 'igmp_icmp_rtp']), 'proto'] = 'others'
    return df


def get_input_output(train, test, label_encoding=False, scaler=None):
    """_summary_

    Args:
        train (dataframe): _description_
        test (dataframe): _description_
        label_encoding (bool, optional): if we want to apply labelencoding, otherwise onehot encoding. Defaults to False.
        scaler (bool, optional): Apply standardscaler for numerical values. Defaults to None.

    Returns:
        x_train, x_test, y_train, y_test: returns  scaled, splitted, and labelencoded(OHE) input and output variables
    """
    x_train, y_train = train.drop(['label'], axis=1), train['label']
    x_test, y_test = test.drop(['label'], axis=1), test['label']
    
    x_train, x_test = feature_engineer(x_train), feature_engineer(x_test)
    
    # Getting categorical columns for x_train from custom function
    categorical_columns = get_categorical_columns(x_train)
    # Using list apprehension for appending columns that are not in the categorical_columns list
    non_categorical_columns = [col for col in x_train.columns if col not in categorical_columns]
    
    # applying StandardScaler for non categorical columns
    if scaler is not None:
        x_train[non_categorical_columns] = scaler.fit_transform(x_train[non_categorical_columns])
        x_test[non_categorical_columns] = scaler.transform(x_test[non_categorical_columns])
        
    if label_encoding:
        x_train, x_test = label_encode(x_train, x_test)
        features = x_train.columns
    else:
        x_train = pd.get_dummies(x_train)
        x_test = pd.get_dummies(x_test)
        # print("Column mismatch {0}, {1}".format(set(x_train.columns)- set(x_test.columns),  set(x_test.columns)- set(x_train.columns)))
        features = list(set(x_train.columns) & set(x_test.columns))
        
    print(f"Number of features {len(features)}")
    x_train = x_train[features]
    x_test = x_test[features]

    return x_train, y_train, x_test, y_test


def display_feature_importance(importance, columns):
    """Create a new dataframe and show importance score for each column.

    Args:
        importance (float): value of importance
        columns (dataframe cols): _description_

    Returns:
        dataframe: Dataframe with scores.
    """
    feature_importance = pd.DataFrame(zip(columns, importance), columns=['Feature', 'Importance'])
    feature_importance['Importance'] /= feature_importance['Importance'].sum()*0.01
    return feature_importance.sort_values(by="Importance", ascending=False)


# ## (3) Data preperation

# In[3]:


train, test = get_train_test()
categorical_columns = get_categorical_columns(train)


# In[4]:


folds = 10
seed = 1
# num_round = 2000
kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed, )
X, Y, x_test, y_test = get_input_output(
    train, 
    test, 
    label_encoding=True, 
    scaler= StandardScaler()
    )
importance_dict = {
    "feature": X.columns
}


# In[5]:


importance_dict


# ## (4) Training Data (using Random Forest Classifier)

# In[6]:


clf = RandomForestClassifier(random_state=1)
clf.fit(X, Y)
feature_importance = clf.feature_importances_
importance_dict['train'] =  feature_importance


# ### Ten-fold Cross Validation

# In[7]:


feature_importances = []

for tr_idx, val_idx in tqdm(kf.split(X, Y), total=folds):
    x_train, y_train = X.iloc[tr_idx], Y[tr_idx]
    # x_val, y_val = X.iloc[val_idx], Y[val_idx]
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    
    feature_importances.append(clf.feature_importances_)

feature_importance = np.mean(feature_importances, axis=0)
importance_dict['train_10_fold'] =  feature_importance
importance_dict
# display_feature_importance(feature_importance, X.columns)


# ## (5) Testing and Training Data (using Random Forest Classifier)

# In[8]:


x_total, y_total = pd.concat([X, x_test]), pd.concat([Y, y_test])


# In[9]:


clf = RandomForestClassifier()
clf.fit(x_total, y_total)
feature_importance = clf.feature_importances_
importance_dict['combined'] =  feature_importance
importance_dict
# display_feature_importance(feature_importance, X.columns)


# ### Ten-fold Cross Validation

# In[10]:


feature_importances = []

for tr_idx, val_idx in tqdm(kf.split(x_total, y_total), total=folds):
    x_train, y_train = x_total.iloc[tr_idx], y_total.iloc[tr_idx]
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    
    feature_importances.append(clf.feature_importances_)

feature_importance = np.mean(feature_importances, axis=0)
importance_dict['combined_10_fold'] =  feature_importance
importance_dict
# display_feature_importance(feature_importance, X.columns)


# In[11]:


importance_df = pd.DataFrame(importance_dict)
for col in importance_df.columns:
    if col=='feature':
        continue
    importance_df[col] = importance_df[col]*100/importance_df[col].sum()
       
importance_df['mean'] = importance_df[[col for col in importance_df.columns if col!='feature']].mean(axis=1)
importance_df = importance_df.sort_values('train_10_fold', ascending=False)
importance_df


# ### (6) Saving the file

# In[12]:


importance_df.to_csv("feature_importance.csv", index=False)
importance_df


# In[13]:


importance_df_round = pd.read_csv("feature_importance.csv",index_col=False)
importance_df_round = importance_df_round.round(7)
importance_df_round


# In[14]:


importance_df_round.to_csv("feature_importance_rounded.csv", index=False)


# In[ ]:




