# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 12:14:47 2023

@author: eriks
"""

# In[1]:
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# metrics are used to find accuracy or error
from sklearn import metrics

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier

# creating instance of labelencoder
encoder = LabelEncoder()
scaler = StandardScaler()   
dt = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# In[7]:
# Models
knn = KNeighborsClassifier(n_neighbors=3)

SFS_model = SequentialFeatureSelector(knn, direction="backward")
ETC_model = ExtraTreesClassifier()
DT_model = DecisionTreeClassifier()

# In[2]:
# Functions
def feature_selectiom_model(model_u, x_tr, y_tr, xx):
# fit the model
    model_u.fit(x_tr, np.ravel(y_tr))
    importance = model_u.feature_importances_

    from matplotlib import pyplot
    import math
    # plot feature importance
    threshold = math.pow(10,-3)

    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.title(label = "Feature importance with threshold: " + str(threshold))
    pyplot.show() 

    this_dict = {}
    remove = {}
    index = set()
    
    for x in range(0, len(importance)):
        #Add feature based on threshold
        if np.array(importance)[x] > threshold:
            this_dict[x] = np.array(importance)[x]
        else:
            remove[x] = np.array(importance)[x]
            index.add(x_tr.columns[x])
    res = {key: val for key, val in sorted(this_dict.items(), key = lambda ele: ele[1], reverse = True)}
    if len(index) > 0:
        x_d_tr = x_tr.copy()
        x_d_tr.drop(labels = index, axis=1, inplace = True)
        x_d_te = xx.copy()
        x_d_te.drop(labels = index, axis=1, inplace = True)
        return res, remove, x_d_tr, x_d_te
    else:
        return res, remove, x_tr , xx

# Feature importance 
def feature_importance(model_list):
    """
    # Best features ranked
    for x in range(0, (len(model_list[0]))):
        print("Column:" + str(list(model_list[0].keys())[x]) + ". " + train.columns[list(model_list[0].keys())[x]])
    """
    if len(model_list[1]) < 0:
        print("No feature removed")
        print("Current shape: ", model_list[2].shape)
    else:
        print("\nFeatures removed\n")
        for x in range(0, len(model_list[1])):
            print("Column:" + str(list(model_list[1].keys())[x]) + ". " + train.columns[list(model_list[1].keys())[x]])
        print("New shape: ", model_list[2].shape)
        
def sampling_strat(sm, x, y):    
    #sm = SMOTE(strategy = "minority",random_state=42)
    #sm = RandomOverSampler(sampling_strategy = "auto",random_state=42)

    x_train_res, y_train_res = sm.fit_resample(x, y)

    return x_train_res, y_train_res


def score_ev(model_s, x, y, xx, yy ,res, name):
    #print(f"\nScoring {len(x.columns)} features for model: {name}")
    scores_tr = cross_val_score(model_s, x, np.ravel(y),scoring = 'accuracy', cv=dt, n_jobs=-1)
    scores_te = cross_val_score(model_s, xx, np.ravel(yy),scoring = 'accuracy', cv=dt, n_jobs=-1)    
    
    if res == True:
        print("Mean train acc value on res: " +  str(np.mean(scores_tr)))
        print("Mean test acc value on res: " +  str(np.mean(scores_te)))
    else:
        print("Mean train acc value: " +  str(np.mean(scores_tr)))
        print("Mean test acc value: " +  str(np.mean(scores_te)))
        
        
def print_Time(t1,s1,t2,s2):
    if t1>t2:
        print(f"{s2} is {t1 - t2:.2f} seconds more efficient than {s1}")
    else:
        print(f"{s1} is {t2 - t1:.2f} seconds more efficient than {s2}")
# In[3]:

train = pd.read_csv('../Dataset/UNSW_NB15_training-set.csv')
test = pd.read_csv('../Dataset/UNSW_NB15_testing-set.csv')

train.drop(['label', 'id'], axis=1, inplace=True)
test.drop(['label', 'id'], axis=1, inplace=True)

# In[4]:

# Defining an empty list
categorical = []
# Iterating through the columns and checking for columns with datatyp "Object"
for col in train.columns:
    if train[col].dtype == 'object' and col != 'attack_cat':
        categorical.append(col) # appending "object" columns to categorical
print(categorical)
# In[]

train.loc[~train['proto'].isin(['tcp', 'udp','arp', 'ospf', 'igmp_icmp_rtp']), 'proto'] = 'others'
test.loc[~test['proto'].isin(['tcp', 'udp','arp', 'ospf', 'igmp_icmp_rtp']), 'proto'] = 'others'

train.loc[~train['state'].isin(['FIN', 'INT', 'CON', 'REQ']), 'state'] = 'others'
test.loc[~test['state'].isin(['FIN', 'INT', 'CON', 'REQ']), 'state'] = 'others'

train.loc[~train['service'].isin(['-', 'dns','http', 'smtp', 'ftp','ftp-data']), 'service'] = 'others'
test.loc[~test['service'].isin(['-', 'dns','http', 'smtp', 'ftp','ftp-data']), 'service'] = 'others'

# In[5]:
x_train, y_train = train.drop(['attack_cat'], axis=1), train['attack_cat']
x_test, y_test = test.drop(['attack_cat'], axis=1), test['attack_cat']
# In[6]:
for x in categorical:
    x_train[x] = encoder.fit_transform(x_train[x])
    x_test[x] = encoder.fit_transform(x_test[x])
   
y_train = encoder.fit_transform(y_train)    
y_test = encoder.fit_transform(y_test)  

# In[]    
x_train.to_csv("../Dataset/Used/xtrain_42FE.csv", index = False)

y_train = pd.DataFrame(y_train, columns=['attack_cat'])
y_train.to_csv("../Dataset/Used/ytrain_42FE.csv", index = False)

x_test.to_csv("../Dataset/Used/xtest_42FE.csv", index = False)

y_test = pd.DataFrame(y_test, columns=['attack_cat'])
y_test.to_csv("../Dataset/Used/ytest_42FE.csv", index = False)  


# In[8]:
"""
score_ev(ETC_model, x_train, y_train, x_test, y_test , False, 'ETC clean')
score_ev(DT_model, x_train, y_train, x_test , y_test, False, 'DT clean')
"""
# In[]

# Function run
#print("Decision Tree\n")
#DT = feature_selectiom_model(DT_model, x_train, y_train, x_test)
#feature_importance(DT)
#print("\nExtraTree classifier\n")
#ETC = feature_selectiom_model(ETC_model, x_train, y_train, x_test)
#feature_importance(ETC)
# In[9]:
#score_ev(ETC_model, ETC[2], y_train, ETC[3], y_test , False, 'ETC FE on clean')
#score_ev(DT_model, DT[2], y_train, DT[3] , y_test, False, 'DT FE on clean')

# In[10]:

dec = 0.5

y_train.value_counts()

val2 = {7: int(np.round(y_train[y_train['attack_cat'] == 7].value_counts() * dec,0)),
        6: int(np.round(y_train[y_train['attack_cat'] == 6].value_counts() * dec,0)),
        5: int(np.round(y_train[y_train['attack_cat'] == 5].value_counts() * dec,0)),
        4: int(np.round(y_train[y_train['attack_cat'] == 4].value_counts() * dec,0)),
        3: int(np.round(y_train[y_train['attack_cat'] == 3].value_counts() * dec,0)),
        2: int(np.round(y_train[y_train['attack_cat'] == 2].value_counts() * dec,0)),
     }
 

val = { 9: int(y_train[y_train['attack_cat'] == 9].value_counts()) * 110,
        8: int(y_train[y_train['attack_cat'] == 8].value_counts()) * 20,
        1: int(y_train[y_train['attack_cat'] == 1].value_counts()) * 12,
        0: int(y_train[y_train['attack_cat'] == 0].value_counts()) * 10,
     }

smote = SMOTE(sampling_strategy = 'minority', random_state=42) # sampling_strategy = "minority", sampling_strategy = val
rus = RandomUnderSampler(sampling_strategy = val2, random_state=42)

# In[11]:
# Resample test

# Accuracy evaluation
#x_res_e, y_res_e = sampling_strat(smote, x_train, y_train)
#x_tres_e, y_tres_e = sampling_strat(smote, x_test, y_test)
"""
print("\nTesting DT...")
#score_ev(DT_model, x_train, y_train, False)
score_ev(DT_model, x_res_e, y_res_e, x_tres_e,y_tres_e, True, "DT clean resample")
print("\nTesting ETC...")
#score_ev(ETC_model, x_train, y_train, False)
score_ev(ETC_model, x_res_e, y_res_e, x_tres_e,y_tres_e, True, 'ETC clean resample')
"""
# In[12]:
#print("\n---Decision Tree---")
#DT_res = feature_selectiom_model(DT_model, x_res_e, y_res_e, x_tres_e)
#feature_importance(DT_res)
"""
print("\n---ExtraTree classifier---")
print(x_res_e.shape)
ETC_res = feature_selectiom_model(ETC_model, x_res_e, y_res_e, x_tres_e)
feature_importance(ETC_res)
print(ETC_res[3].shape)
"""
# In[]
# Testing samples
"""
score_ev(ETC_model, ETC_res[2], y_res_e, ETC_res[3], y_tres_e, True, 'ETC FE resample')
score_ev(DT_model, DT_res[2], y_res_e, DT_res[3], y_tres_e, True, 'DT FE resample')
"""
# In[13]:
# Correlation check
Cors = set()
cor_mat = x_train.corr()

# https://journals.sagepub.com/doi/pdf/10.1177/875647939000600106
cor_t = 0.68

for i in range(len(cor_mat .columns)):
    for j in range(i):
        if abs(cor_mat.iloc[i, j]) > cor_t:
            colname = cor_mat.columns[i]
            Cors.add(colname)

x_train_cor = x_train.copy()
x_train_cor.drop(labels = Cors, axis=1, inplace = True)

x_test_cor = x_test.copy()
x_test_cor.drop(labels = Cors, axis=1, inplace = True)

print(x_train.shape)
print(x_train_cor.shape)
"""
x_train_cor.to_csv(f"../Dataset/xtrain_cor_{x_test_cor.shape[1]}FE.csv", index = False)
x_test_cor.to_csv(f"../Dataset/xtest_cor_{x_test_cor.shape[1]}FE.csv", index = False)

y_test.to_csv(f"../Dataset/ytest_cor_{x_test_cor.shape[1]}FE.csv", index = False)  
y_train.to_csv(f"../Dataset/ytrain_cor_{x_test_cor.shape[1]}FE.csv", index = False)
"""
# In[]
"""
score_ev(ETC_model, x_train_cor, y_train, x_test_cor, y_test, False ,"ETC cor")
score_ev(DT_model, x_train_cor, y_train, x_test_cor, y_test, False ,"DT cor")
"""
# In[14]:
print("ETC with correlation features\n")
ETC_cor = feature_selectiom_model(ETC_model, x_train_cor, y_train, x_test_cor)
ETC_cor_fi = feature_importance(ETC_cor)

#print("\nDT with correlation features\n")
#DT_cor = feature_selectiom_model(DT_model, x_train_cor, y_train, x_test_cor)
#DT_cor_fi = feature_importance(DT_cor)

"""
print("\nEvaluation..")
score_ev(ETC_model, ETC_cor[2], y_train, ETC_cor[3], y_test, False ,"ETC FE cor")
score_ev(DT_model, DT_cor[2], y_train, DT_cor[3],y_test, False ,"DT FE cor")
"""
# In[15]:
# In[]
"""
# Scale the data

x_train_etc_cor_scale = scaler.fit_transform(ETC_cor[2])
x_test_etc_cor_scale = scaler.fit_transform(ETC_cor[3])

x_train_dt_cor_scale = scaler.fit_transform(DT_cor[2])
x_test_dt_cor_scale = scaler.fit_transform(DT_cor[3])      
"""  
# Correlation data with SMOTE sampling
xx_cor_etc_samp, yy_cor_etc_samp = sampling_strat(smote, ETC_cor[2], y_train)
x_cor_tt_etc_samp, y_cor_tt_etc_samp = sampling_strat(smote, ETC_cor[3], y_test)




x_cor_etc_samp, y_cor_etc_samp = sampling_strat(rus, xx_cor_etc_samp, yy_cor_etc_samp)
x_cor_t_etc_samp, y_cor_t_etc_samp = sampling_strat(rus, x_cor_tt_etc_samp, y_cor_tt_etc_samp)

print(xx_cor_etc_samp.shape)
print(x_cor_etc_samp.shape)

print(x_cor_tt_etc_samp.shape)
print(x_cor_t_etc_samp.shape)


#x_cor_t_etc_samp, y_cor_t_etc_samp = x_cor_tt_etc_samp, y_cor_tt_etc_samp
#x_cor_etc_samp, y_cor_etc_samp = xx_cor_etc_samp, yy_cor_etc_samp

#x_cor_dt_samp, y_cor_dt_samp = sampling_strat(smote, DT_cor[2], y_train)
print("\nETC scoring: ")
score_ev(ETC_model, x_cor_etc_samp, y_cor_etc_samp, x_cor_t_etc_samp , y_cor_t_etc_samp, True, "ETC with cor resample")
#print("\nDT scoring: ")
#score_ev(DT_model, x_cor_dt_samp, y_cor_dt_samp, x_cor_dt_samp, y_cor_dt_samp, True, "DT with cor resample")

# In[16]:
""" 
# Testing complexity
import time
# Regular model
start = time.time()
score_ev(ETC_model, x_train, y_train, False, "ETC")
end = time.time()
reg_time = end-start
print(f'Runtime: {reg_time:.2f} seconds')
# Resample model
start = time.time()
score_ev(ETC_model, x_res, y_res, True, "ETC resample")
end = time.time()
samp_time = end-start
print(f'Runtime: {samp_time:.2f} seconds')
# Correlation
start = time.time()
score_ev(ETC_model, x_train_cor, y_train, True, "ETC cor")
end = time.time()
cor_time = end-start
print(f'Runtime: {cor_time:.2f} seconds')
# Correlation with etc feature selection
start = time.time()
score_ev(ETC_model, ETC_cor[2], y_train, True, "ETC cor with FS")
end = time.time()
cor_etc_time = end-start
print(f'Runtime: {cor_time:.2f} seconds')
# Correlation with resample model
start = time.time()
score_ev(ETC_model, x_cor_etc_samp, y_cor_etc_samp, True, "ETC cor FS resamp")
end = time.time()
cor_etc_samp_time = end-start
print(f'Runtime: {cor_etc_samp_time:.2f} seconds')

# In[]
# Efficiacy      
print("\nBest model:")
print_Time(reg_time,"Regular", samp_time, "Sample")
print_Time(cor_time, "Corr" ,reg_time, "Regular")
print_Time(cor_time, "Corr" , samp_time, "Sample")
print("///")
print_Time(cor_etc_time, "Cor_etc" ,cor_time, "cor")
print_Time(cor_etc_samp_time, "Cor_etc_samp" ,cor_time, "cor")
print_Time(cor_etc_samp_time, "Cor_etc_samp" ,cor_etc_time, "cor_etc")
"""
# In[]
# Saving best to csv
xtrain = f'../Dataset/Used/xtrain_{x_cor_etc_samp.shape[1]}FE.csv'
ytrain = f'..//Dataset/Used/ytrain_{x_cor_etc_samp.shape[1]}FE.csv'

x_train_df = pd.DataFrame(x_cor_etc_samp)
x_train_df.to_csv(xtrain, index = False)

y_train_df = pd.DataFrame(y_cor_etc_samp, columns=['attack_cat'])
y_train_df.to_csv(ytrain, index = False)

xtest = f'../Dataset/Used/xtest_{ETC_cor[3].shape[1]}FE.csv'
ytest = f'../Dataset/Used/ytest_{ETC_cor[3].shape[1]}FE.csv'

x_test_df = pd.DataFrame(x_cor_t_etc_samp)
x_test_df.to_csv(xtest, index = False)

y_test_df = pd.DataFrame(y_cor_t_etc_samp, columns=['attack_cat'])
y_test_df.to_csv(ytest, index = False)

print(x_train_df.shape)
print(x_test_df.shape)
