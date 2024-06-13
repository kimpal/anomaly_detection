# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 12:14:47 2023

@author: eriks
"""

# In[1]:
import warnings
import time


warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import LabelEncoder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler as MS
from sklearn.ensemble import ExtraTreesClassifier

# creating instance of labelencoder
encoder = LabelEncoder()   
dt = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# In[7]:
# Models

ETC_model = ExtraTreesClassifier()
  
d = 3  
t = 2


if d == 1:
    Dataset = "UNSW_NB15" # 21 42 m 22 42 m
if d == 2:
    Dataset = "TON_Train_Test" # 17 41 mb
if d == 3:
    Dataset = "IoT_Botnet" # 12 16 m - 9 13 b



score = False
save = False
Time_complex = False

if t == 1:
    TYPE = "Binary"
if t == 2:
    TYPE = "Multi" #Binary #Multi


if Dataset == "TON_Train_Test":
    df = pd.read_csv(f'../Dataset/{Dataset}.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    df['type'].value_counts()
    train = df.iloc[:int(np.round(len(df)*0.70)), :]
    train.reset_index()
    test = df.iloc[int(np.round(len(df)*0.70))+1:, :]
    test.reset_index()
    print("Train: ", len(np.unique(train['type'])), " ", train.shape) 
    print("Test: ", len(np.unique(test['type'])), " " , test.shape)   
if Dataset == "UNSW_NB15":
    train = pd.read_csv(f'../Dataset/{Dataset}_Test.csv')
    test = pd.read_csv(f'../Dataset/{Dataset}_Train.csv')
if Dataset == "IoT_Botnet":
    train = pd.read_csv(f'../Dataset/{Dataset}_Train.csv')
    test = pd.read_csv(f'../Dataset/{Dataset}_Test.csv')

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
    
    # Best features ranked
    for x in range(0, (len(model_list[0]))):
        print("Column:" + str(list(model_list[0].keys())[x]) + ". " + train.columns[list(model_list[0].keys())[x]])
    
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
    scores_tr = cross_val_score(model_s, x, np.ravel(y),scoring = 'accuracy', cv=dt, n_jobs=1)
    scores_te = cross_val_score(model_s, xx, np.ravel(yy),scoring = 'accuracy', cv=dt, n_jobs=1)    
    
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

Main_feat = 0

if TYPE == "Multi":
    if Dataset == "UNSW_NB15":
        train.drop(['label', 'id'], axis=1, inplace=True)
        test.drop(['label', 'id'], axis=1, inplace=True)
        
        Main_feat = "attack_cat"
        
        
    if Dataset == "IoT_Botnet":   # pkSeqID, saddr, daddr, proto, state, and flgs
        train.drop(['attack'], axis=1, inplace=True)
        test.drop(['attack'], axis=1, inplace=True)
        
        train.drop(['drate'], axis=1, inplace=True)
        test.drop(['drate'], axis=1, inplace=True)
        
        Main_feat = "category"
    
    if Dataset == "TON_Train_Test":
        train.drop(['label'], axis=1, inplace=True)
        test.drop(['label'], axis=1, inplace=True)
        
        train.drop(['src_ip'], axis=1, inplace=True)
        test.drop(['src_ip'], axis=1, inplace=True)
        
        train.drop(['dst_ip'], axis=1, inplace=True)
        test.drop(['dst_ip'], axis=1, inplace=True)
        
        Main_feat = "type"
        
else:
    if Dataset == "UNSW_NB15":
        train.drop(['attack_cat', 'id'], axis=1, inplace=True)
        test.drop(['attack_cat', 'id'], axis=1, inplace=True)
        
        Main_feat = "label"

        
    if Dataset == "IoT_Botnet":
        train.drop(['category'], axis=1, inplace=True)
        test.drop(['category'], axis=1, inplace=True)
        
        train.drop(['drate'], axis=1, inplace=True)
        test.drop(['drate'], axis=1, inplace=True)
        
        train.drop(['pkSeqID'], axis=1, inplace=True)
        test.drop(['pkSeqID'], axis=1, inplace=True)
        
        train.drop(['saddr'], axis=1, inplace=True)
        test.drop(['saddr'], axis=1, inplace=True)
        
        train.drop(['daddr'], axis=1, inplace=True)
        test.drop(['daddr'], axis=1, inplace=True)
        
        #train.drop(['proto'], axis=1, inplace=True)
        #test.drop(['proto'], axis=1, inplace=True)
        
         
        Main_feat = "attack"

        
    if Dataset == "TON_Train_Test":
        train.drop(['type'], axis=1, inplace=True)
        test.drop(['type'], axis=1, inplace=True)
        
        train.drop(['src_ip'], axis=1, inplace=True)
        test.drop(['src_ip'], axis=1, inplace=True)
        
        train.drop(['dst_ip'], axis=1, inplace=True)
        test.drop(['dst_ip'], axis=1, inplace=True)
        
        Main_feat = "label"

  
# In[4]:    
# Defining an empty list
categorical = []
# Iterating through the columns and checking for columns with datatyp "Object"
for col in train.columns:
    if train[col].dtype == 'object' and (col != Main_feat):
        categorical.append(col) # appending "object" columns to categorical
print(categorical)
train['proto'].unique()
train['saddr'].unique()
train['sport'].unique()
train['daddr'].unique()
train['dport'].unique()
train['subcategory'].unique()
# In[]

if Dataset == "UNSW_NB15":
    
    train['proto'].value_counts()
    train['state'].value_counts()
    train['service'].value_counts()
    
    train.loc[~train['proto'].isin(['tcp', 'udp','arp', 'ospf', 'igmp_icmp_rtp']), 'proto'] = 'others'
    test.loc[~test['proto'].isin(['tcp', 'udp','arp', 'ospf', 'igmp_icmp_rtp']), 'proto'] = 'others'

    train.loc[~train['state'].isin(['FIN', 'INT', 'CON', 'REQ']), 'state'] = 'others'
    test.loc[~test['state'].isin(['FIN', 'INT', 'CON', 'REQ']), 'state'] = 'others'

    train.loc[~train['service'].isin(['-', 'dns','http', 'smtp', 'ftp','ftp-data']), 'service'] = 'others'
    test.loc[~test['service'].isin(['-', 'dns','http', 'smtp', 'ftp','ftp-data']), 'service'] = 'others'

# In[5]:
    
x_train, y_train = train.drop([Main_feat], axis=1), train[Main_feat]
x_test, y_test = test.drop([Main_feat], axis=1), test[Main_feat]

# In[6]:


for x in categorical:
    x_train[x] = encoder.fit_transform(x_train[x])
    x_test[x] = encoder.fit_transform(x_test[x])
   
y_train = encoder.fit_transform(y_train)    
y_test = encoder.fit_transform(y_test)  

#scaler = MS()
#X_train = scaler.fit_transform(x_train)
#X_test = scaler.transform(x_test)

# In[]    

print("Train shape: ", x_train.shape )
print("Test shape: ", x_test.shape)  

y_train = pd.DataFrame(y_train, columns=[Main_feat])
y_test = pd.DataFrame(y_test, columns=[Main_feat])

if save == True:
    x_train.to_csv(f"../Dataset/Used/{Dataset}_xtrain_{x_train.shape[1]}FE_{TYPE}.csv", index = False)
    x_test.to_csv(f"../Dataset/Used/{Dataset}_xtest_{x_test.shape[1]}FE_{TYPE}.csv", index = False)
        

    y_train.to_csv(f"../Dataset/Used/{Dataset}_ytrain_{x_train.shape[1]}FE_{TYPE}.csv", index = False)
    
   
    y_test.to_csv(f"../Dataset/Used/{Dataset}_ytest_{x_test.shape[1]}FE_{TYPE}.csv", index = False)  

# In[8]:
#score_ev(ETC_model, x_train, y_train, x_test, y_test , False, 'ETC clean')

#etc = feature_selectiom_model(ETC_model, x_train, y_train, x_test)

#etc[2].shape

#score_ev(ETC_model, etc[2], y_train, etc[3], y_test, False, "ETC FE")

# In[11]
dec = 0.5

if TYPE == "Multi" and Dataset == "UNSW_NB15":
   
    val2_test = {
            #9: int(np.round(y_test[y_test[Main_feat] == 6].value_counts() * dec,0)),
            7: int(np.round(y_test[y_test[Main_feat] == 7].value_counts() * dec,0)),
            6: int(np.round(y_test[y_test[Main_feat] == 6].value_counts() * dec,0)),
            5: int(np.round(y_test[y_test[Main_feat] == 5].value_counts() * dec,0)),
            4: int(np.round(y_test[y_test[Main_feat] == 4].value_counts() * dec,0)),
            3: int(np.round(y_test[y_test[Main_feat] == 3].value_counts() * dec,0)),
            2: int(np.round(y_test[y_test[Main_feat] == 2].value_counts() * dec,0)),
   }
    
    val2_train = {
            #9: int(np.round(y_train[y_train[Main_feat] == 6].value_counts() * dec,0)),
            7: int(np.round(y_train[y_train[Main_feat] == 7].value_counts() * dec,0)),
            6: int(np.round(y_train[y_train[Main_feat] == 6].value_counts() * dec,0)),
            5: int(np.round(y_train[y_train[Main_feat] == 5].value_counts() * dec,0)),
            4: int(np.round(y_train[y_train[Main_feat] == 4].value_counts() * dec,0)),
            3: int(np.round(y_train[y_train[Main_feat] == 3].value_counts() * dec,0)),
            2: int(np.round(y_train[y_train[Main_feat] == 2].value_counts() * dec,0)),
            }
  
    smote_train = SMOTE(sampling_strategy = "minority", random_state=42) # sampling_strategy = "minority", sampling_strategy = val
    rus_train = RandomUnderSampler(sampling_strategy = val2_test, random_state=42)

    smote_test = SMOTE(sampling_strategy = "minority", random_state=42) # sampling_strategy = "minority", sampling_strategy = val
    #rus_test = RandomUnderSampler(sampling_strategy = val2_test, random_state=42)   #val2_train: 90.3     #val2_test: 87.5  all 0.5
    
    
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

print(x_train_cor.shape)
print(x_test_cor.shape)

xtrain = f'../Dataset/Used/{Dataset}_xtrain_{x_train_cor.shape[1]}FE_{TYPE}_cor.csv'
ytrain = f'..//Dataset/Used/{Dataset}_ytrain_{x_train_cor.shape[1]}FE_{TYPE}_cor.csv'
        

x_train_df = pd.DataFrame(x_train_cor)
x_train_df.to_csv(xtrain, index = False)
    
y_train_df = pd.DataFrame(y_train, columns=[Main_feat])
y_train_df.to_csv(ytrain, index = False)


xtest = f'../Dataset/Used/{Dataset}_xtest_{x_test_cor.shape[1]}FE_{TYPE}_cor.csv'
ytest = f'..//Dataset/Used/{Dataset}_ytest_{x_test_cor.shape[1]}FE_{TYPE}_cor.csv'
        

x_test_df = pd.DataFrame(x_test_cor)
x_test_df.to_csv(xtest, index = False)
    
y_test_df = pd.DataFrame(y_test, columns=[Main_feat])
y_test_df.to_csv(ytest, index = False)


print(x_train_cor.shape)
print(x_test_cor.shape)

#score_ev(ETC_model, x_train_cor, y_train, x_test_cor, y_test, False, "cor clean")
# In[12]:

if Dataset == "UNSW_NB15" and TYPE == "Multi":
    xx_cor_etc_samp, yy_cor_etc_samp = sampling_strat(smote_train, x_train_cor, y_train)
    x_cor_etc_samp, y_cor_etc_samp = sampling_strat(rus_train, xx_cor_etc_samp, yy_cor_etc_samp)
    
    x_cor_t_etc_samp, y_cor_t_etc_samp = sampling_strat(smote_test, x_test_cor, y_test)
    #x_cor_t_etc_samp, y_cor_t_etc_samp = sampling_strat(rus_test, x_cor_tt_etc_samp, y_cor_tt_etc_samp)  

    print("Train data")
    print("Start: ", x_train.shape)
    print("Samp: ", xx_cor_etc_samp.shape)
    print("Reduction: ", x_cor_etc_samp.shape)
    print("Reduction: ", y_cor_etc_samp.value_counts())
    print("Test data")
    print(y_cor_t_etc_samp.shape)
else:
    x_cor_etc_samp, y_cor_etc_samp = x_train_cor, y_train
    x_cor_t_etc_samp, y_cor_t_etc_samp = x_test_cor, y_test
    
# In[14]:
    
print("ETC with correlation features\n")
ETC_cor = feature_selectiom_model(ETC_model, x_cor_etc_samp, y_cor_etc_samp, x_cor_t_etc_samp)
ETC_cor_fi = feature_importance(ETC_cor)

start = time.time()
#score_ev(ETC_model, ETC_cor[2], y_cor_etc_samp, ETC_cor[3], y_cor_t_etc_samp, False, "Smote b4 Cor FE")
end = time.time()
print("Runtime: " + str(end-start))

print(ETC_cor[2].shape)
print(ETC_cor[3].shape)
# In[15]:
# Correlation data with SMOTE sampling

if score:
    if Dataset == "UNSW_NB15" and TYPE == "Multi":
        print("\nETC scoring: ")
        score_ev(ETC_model, x_cor_etc_samp, y_cor_etc_samp, ETC_cor[3] , y_test, True, "ETC with cor")
    else:
        print("\nETC scoring: ")
        score_ev(ETC_model, ETC_cor[2], y_train, ETC_cor[3] , y_test, True, "ETC with cor")
#print("\nDT scoring: ")
#score_ev(DT_model, x_cor_dt_samp, y_cor_dt_samp, x_cor_dt_samp, y_cor_dt_samp, True, "DT with cor resample")
# In[16]:

# Testing complexity
if Time_complex == True:

    
    # Regular model
    print("Starting evaluation on time complexity\n")
    start = time.time()
    score_ev(ETC_model, x_train, y_train, x_test, y_test ,False, "ETC")
    end = time.time()
    reg_time = end-start
    print(f'Clean Runtime: {reg_time:.2f} seconds')
    
    # Correlation
    start = time.time()
    score_ev(ETC_model, x_train_cor, y_train, x_test_cor , y_test, True, "ETC cor")
    end = time.time()
    cor_time = end-start
    print(f'Cor Runtime: {cor_time:.2f} seconds')
    
    # Correlation with resample model
    start = time.time()
    score_ev(ETC_model, x_cor_etc_samp, y_cor_etc_samp, ETC_cor[3], y_test, True, "ETC cor FS resamp")
    end = time.time()
    cor_etc_samp_time = end-start
    print(f'Cor FE resample Runtime: {cor_etc_samp_time:.2f} seconds')
    
    """
    # Regular model
    print("Starting evaluation on time complexity\n")
    start = time.time()
    score_ev(ETC_model, x_train, y_train, x_test, y_test ,False, "ETC")
    end = time.time()
    reg_time = end-start
    print(f'Clean Runtime: {reg_time:.2f} seconds')
    # Resample model
    start = time.time()
    score_ev(ETC_model, xx_res_e, yy_res_e, x_test , y_test, True, "ETC resample")
    end = time.time()
    samp_time = end-start
    print(f'Clean resample Runtime: {samp_time:.2f} seconds')
    # Resample model no RuS
    start = time.time()
    score_ev(ETC_model, x_res_e, y_res_e, x_test , y_test, True, "ETC resample")
    end = time.time()
    samp_time = end-start
    print(f'Clean resample no RuS Runtime: {samp_time:.2f} seconds\n')
    
    
    # FE model
    start = time.time()
    score_ev(ETC_model, etc[2], y_train, etc[3], y_test ,False, "ETC")
    end = time.time()
    reg_time = end-start
    print(f'FE Runtime: {reg_time:.2f} seconds')
    
    # FE resample model no RuS
    x_res_fe, y_res_fe = sampling_strat(smote_train, etc[2], y_train)
    xx_res_fe, yy_res_fe = sampling_strat(smote_train, etc[3], y_test)
    start = time.time()
    score_ev(ETC_model, x_res_fe, y_res_fe, xx_res_fe, yy_res_fe ,False, "ETC")
    end = time.time()
    reg_time = end-start
    print(f'FE resample no RuS Runtime: {reg_time:.2f} seconds')
    
    # FE model resample
    xx, yy = sampling_strat(rus_train, x_res_fe, y_res_fe)
    #xxx, yyy = sampling_strat(rus_test, xx_res_fe, yy_res_fe)
    start = time.time()
    score_ev(ETC_model, xx, yy, x_test, y_test ,False, "ETC")
    end = time.time()
    reg_time = end-start
    print(f'FE resample Runtime: {reg_time:.2f} seconds\n')
    
    
    # Correlation
    start = time.time()
    score_ev(ETC_model, x_train_cor, y_train, x_test_cor , y_test, True, "ETC cor")
    end = time.time()
    cor_time = end-start
    print(f'Cor Runtime: {cor_time:.2f} seconds')
    
    # Correlation resample no RuS
    x_res_fe, y_res_fe = sampling_strat(smote_train, x_train_cor, y_train)
    xx_res_fe, yy_res_fe = sampling_strat(smote_train, x_test_cor, y_test)
    start = time.time()
    score_ev(ETC_model, x_res_fe, y_res_fe, xx_res_fe , yy_res_fe, True, "ETC cor")
    end = time.time()
    cor_time = end-start
    print(f'Cor resample no RuS Runtime: {cor_time:.2f} seconds')
   
    # Correlation resample
    xx, yy = sampling_strat(rus_train, x_res_fe, y_res_fe)
    #xxx, yyy = sampling_strat(rus_test, xx_res_fe, yy_res_fe)
    start = time.time()
    score_ev(ETC_model, xx, yy, ETC_cor[3] , y_test, True, "ETC cor")
    end = time.time()
    cor_time = end-start
    print(f'Cor resample Runtime: {cor_time:.2f} seconds\n')     
    
    # Correlation with etc feature selection
    start = time.time()
    score_ev(ETC_model, ETC_cor[2], y_train, ETC_cor[3], y_test, True, "ETC cor with FS")
    end = time.time()
    cor_etc_time = end-start
    print(f'Cor with FE Runtime: {cor_time:.2f} seconds')
    
    # Correlation with resample model (NO RuS)
    start = time.time()
    score_ev(ETC_model, xx_cor_etc_samp, yy_cor_etc_samp, x_test_cor, y_test, True, "ETC cor FS resamp")
    end = time.time()
    cor_etc_samp_time = end-start
    print(f'Cor FE resample no RuS Runtime: {cor_etc_samp_time:.2f} seconds\n')
    
    # Correlation with resample model
    start = time.time()
    score_ev(ETC_model, x_cor_etc_samp, y_cor_etc_samp, ETC_cor[3], y_test, True, "ETC cor FS resamp")
    end = time.time()
    cor_etc_samp_time = end-start
    print(f'Cor FE resample Runtime: {cor_etc_samp_time:.2f} seconds')
    y_cor_etc_samp.value_counts()
    
    """
    
# In[]
# Saving best to csv
if save == True:
    if TYPE == "Multi":
    
        print("Saving multiclass dataset")
        xtrain = f'../Dataset/Used/{Dataset}_xtrain_{ETC_cor[2].shape[1]}FE_{TYPE}.csv'
        ytrain = f'..//Dataset/Used/{Dataset}_ytrain_{ETC_cor[2].shape[1]}FE_{TYPE}.csv'
        
        
        x_train_df = pd.DataFrame(ETC_cor[2])
        x_train_df.to_csv(xtrain, index = False)
    
        y_train_df = pd.DataFrame(y_cor_etc_samp, columns=[Main_feat])
        y_train_df.to_csv(ytrain, index = False)
    
        xtest = f'../Dataset/Used/{Dataset}_xtest_{ETC_cor[3].shape[1]}FE_{TYPE}.csv'
        ytest = f'../Dataset/Used/{Dataset}_ytest_{ETC_cor[3].shape[1]}FE_{TYPE}.csv'
    
        x_test_df = pd.DataFrame(ETC_cor[3])#(x_cor_t_etc_samp)
        x_test_df.to_csv(xtest, index = False)
        #y_cor_t_etc_samp
        y_test_df = pd.DataFrame(y_test, columns=[Main_feat])
        y_test_df.to_csv(ytest, index = False)
        
        print(ETC_cor[2].shape)
        print(ETC_cor[3].shape)
       
    else:
         print("Saving binary dataset")
         xtrain = f'../Dataset/Used/{Dataset}_xtrain_{ETC_cor[2].shape[1]}FE_{TYPE}.csv'
         ytrain = f'..//Dataset/Used/{Dataset}_ytrain_{ETC_cor[2].shape[1]}FE_{TYPE}.csv'
    
         x_train_df = pd.DataFrame(ETC_cor[2])
         x_train_df.to_csv(xtrain, index = False)
    
         y_train_df = pd.DataFrame(y_train, columns=[Main_feat])
         y_train_df.to_csv(ytrain, index = False)
    
         xtest = f'../Dataset/Used/{Dataset}_xtest_{ETC_cor[3].shape[1]}FE_{TYPE}.csv'
         ytest = f'../Dataset/Used/{Dataset}_ytest_{ETC_cor[3].shape[1]}FE_{TYPE}.csv'
    
         x_test_df = pd.DataFrame(ETC_cor[3])#(x_cor_t_etc_samp)
         x_test_df.to_csv(xtest, index = False)

        #y_cor_t_etc_samp
         y_test_df = pd.DataFrame(y_test, columns=[Main_feat])   
         y_test_df.to_csv(ytest, index = False)
         print(x_train_df.shape)
         print(x_test_df.shape)
