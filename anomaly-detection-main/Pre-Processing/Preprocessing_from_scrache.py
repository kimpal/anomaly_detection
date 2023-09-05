
# from array import array
# Most of the information on what to convert is from her:https://medium.com/@subrata.maji16/building-an-intrusion-detection-system-on-unsw-nb15-dataset-based-on-machine-learning-algorithm-16b1600996f5
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
# from sklearn.model_selection import train_test_split
# Reading datasets
dfs = []
for i in range(1,5):
    path = '../Dataset/UNSW-NB15_{}.csv'  # There are four (4) input csv files
    dfs.append(pd.read_csv(path.format(i),low_memory=False, header = None))
all_data = pd.concat(dfs).reset_index(drop=True)  # Concat all to a single df


# This csv file contains the names of all the features
df_col = pd.read_csv('../Dataset/NUSW-NB15_features.csv', encoding='ISO-8859-1')

# Making column names lower case, removing spaces
df_col['Name'] = df_col['Name'].apply(lambda x: x.strip().replace(' ', '').lower())
# Renaming our dataframe with proper column names
all_data.columns = df_col['Name']

print(all_data.head())
print(all_data.shape)
#giving index Name the name id
all_data['id'] = all_data.index
#all_data['Name'] = all_data['id'] # tst this
#all_data = all_data.rename(columns={'id': 'Name'})
#all_data['id'] = all_data.index
print(all_data.head())

# splitting dataframe by row index
train = all_data.iloc[:1778032,:]
test = all_data.iloc[:762015,:]
print("Shape of new dataframes - {} , {}".format(train.shape, test.shape))
print(train.shape,'\n',test.shape)
# Save the training and test datasets to CSV files
train.to_csv('../Dataset/train_data.csv', index=False)
test.to_csv('../Dataset/test_data.csv', index=False)

print("-----------------------------------Reading data----------------------------------------------")
# checkint the new saved dataset
train_Cech = pd.read_csv('../Dataset/train_data.csv', low_memory=False)
#test_Cech = pd.read_csv('../Dataset/test_data.csv', low_memory=False)
#print(test_Cech.head())
print(train_Cech.head())
print(train_Cech.shape)
#print(test_Cech.shape)
print(train_Cech.attack_cat.unique())
print(train_Cech.dtypes)
print(train_Cech.isnull().values.any())
print(train_Cech.isnull().any())
# Counting NaN values in all columns
nan_count = train_Cech.isna().sum()
print(nan_count)
print(train_Cech['ct_ftp_cmd'].unique())
print(train_Cech['is_ftp_login'].unique())

print("-------------replacing nan values-----------------------")
# We don’t have “normal” values for “attack_cat,” so we must fill Null values with “normal”
train_Cech['attack_cat'] = train_Cech.attack_cat.fillna(value='normal').apply(lambda x: x.strip().lower())
# replacing nan values of ct_flw_http_mthd and is_ftp_login whit 0
train_Cech[["ct_flw_http_mthd", "is_ftp_login"]] = train_Cech[["ct_flw_http_mthd", "is_ftp_login"]].fillna(0)
train_Cech['ct_flw_http_mthd'] = train_Cech['ct_flw_http_mthd'].apply(np.int64)
train_Cech.attack_cat.value_counts()
print(train_Cech.attack_cat.unique())
nan_count = train_Cech.isna().sum()
print(nan_count)
print(train_Cech.dtypes)

#############------------------################--------------#######
# bolean column whit other number than bolen
train_Cech['is_ftp_login'] = np.where(train_Cech['is_ftp_login']>1, 1, train_Cech['is_ftp_login'])
train_Cech['is_ftp_login'] = train_Cech['is_ftp_login'].apply(np.int64)
## re namong pural atack to sigular
train_Cech['attack_cat'] = train_Cech['attack_cat'].replace('backdoors','backdoor',)
print(train_Cech['attack_cat'].unique())

# In the research paper, it was mentioned that, this is a numerical feature and not categorical
print("---")
print("datatype of ct_ftp_cmd: ",train_Cech['ct_ftp_cmd'].dtypes)
print("unique valuess of ct_ftp_cmd: ",train_Cech['ct_ftp_cmd'].unique())
print("sum off empty string: ",(train_Cech['ct_ftp_cmd'] == '').sum())
print("summ of string whit spaces ' ': ",(train_Cech['ct_ftp_cmd'] == ' ').sum())
# identifying empty strings int the column that nead to be removed in order to convert to int64

print("------------convert ct_ftp_cmd to numerical values by dropping some column-----------------")
#Not the best slulution to tropp convert '' '' to nan adn than drop enter row
# but works for now
train_Cech['ct_ftp_cmd'].replace(' ', np.nan, inplace=True)
train_Cech = train_Cech.dropna()
train_Cech['ct_ftp_cmd'] = train_Cech['ct_ftp_cmd'].apply(np.int64)
print("hopefully converted")
print(train_Cech['ct_ftp_cmd'].unique())
nan_count = train_Cech.isna().sum()
print(nan_count)
print(train_Cech['ct_ftp_cmd'].dtypes)

## re fixing the index after removed values
all_data.reset_index(drop=True)
#df_col['Name'] = df_col['Name'].apply(lambda x: x.strip().replace(' ', '').lower())


print("------------all preprocessing on train done printing results finished printing results------------------")
print('shape of train_cech: ', train_Cech.shape)
print(train_Cech.attack_cat.unique())
print(train_Cech.dtypes)
nan_count = train_Cech.isna().sum()
print(nan_count)
print("---------")
print(train_Cech['ct_ftp_cmd'].unique())
print(train_Cech['ct_ftp_cmd'].dtypes)
print(train_Cech['is_ftp_login'].unique())

"""
print('')
print("---------------comparing to the UNSW_NB15_train-set----------------------")
train = pd.read_csv('../../Anomaly-Detection-main/Dataset/UNSW_NB15_training-set.csv')
test = pd.read_csv('../../Anomaly-Detection-main//Dataset/UNSW_NB15_testing-set.csv')
print('dataset in shape of train: ', train.shape)
print('dataset in shape of tes: ', test.shape)
print(train.attack_cat.unique())
print(train.dtypes)
nan_count = train.isna().sum()
print(nan_count)
print("----------")
print(train['ct_ftp_cmd'].unique())
print(train['ct_ftp_cmd'].dtypes)
print(train['is_ftp_login'].unique())

print("------------UNSW_NB15_test-set--------------")
print(test.attack_cat.unique())
print(test.dtypes)
nan_count = test.isna().sum()
print(nan_count)
print(test['ct_ftp_cmd'].unique())
print(test['ct_ftp_cmd'].dtypes)
print(test['is_ftp_login'].unique())
"""
#the train data is preprocessed let us visualize it
"""
def col_countplot(col, train_data=train_Cech):
    #This function plots countplot of a given feature for train dataset
    fig, ax = plt.subplots(figsize=(8,4))
    sns.set_style('whitegrid')
    # countplot of the given column
    ax = sns.countplot(x=col, hue='label', data=train_data)
    ax.legend(loc="upper right", labels=('normal', 'attack'))
    ax.set_title("train data")
    plt.xticks(rotation=45)
    plt.show()
"""

""""
# Plotting pdf of numerical columns
# Refer: https://www.kaggle.com/khairulislam/unsw-nb15-eda
def dual_plot(col, data1=normal, data2=anomaly, label1='normal', label2='anomaly', method=None):

# This function plots pdf of the given feature on attack and non-attack data

    if method != None:
        sns.set_style('whitegrid')
        sns.distplot(data1[col].apply(method), label=label1, hist=False, rug=True)
        sns.distplot(data2[col].apply(method), label=label2, hist=False, rug=True)
    else:
        sns.set_style('whitegrid')
        sns.distplot(data1[col], label=label1, hist=False, rug=True)
        sns.distplot(data2[col], label=label2, hist=False, rug=True)
    plt.legend()
"""