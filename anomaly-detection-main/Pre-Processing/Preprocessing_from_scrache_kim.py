
# from array import array
# Most of the information on what to convert is from her:https://medium.com/@subrata.maji16/building-an-intrusion-detection-system-on-unsw-nb15-dataset-based-on-machine-learning-algorithm-16b1600996f5
import os.path
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
# from sklearn.model_selection import train_test_split

# defining the files to be checked if exists
name_of_file1 ='train_data'
name_of_file2 ='test_data'
file1 ='../Dataset/'+name_of_file1+'.csv'
file2 ='../Dataset/'+name_of_file2+'.csv'
#file1 = '../Dataset/train_data.csv'
#file2 = '../Dataset/test_data.csv'


# function to check for file existence returning boolean values
def existing_files(file_name):
    file_exist = os.path.isfile(file_name)
    print("file: ", file_name, "Exists:", file_exist)
    return file_exist


def create_train_test():
    print("create_train_test()", "is running...")
    # Reading UNSW-NB15 datasets
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
    # Save the training and test datasets to CSV files if it not already exists
    train.to_csv(file1, index=False)
    test.to_csv(file2, index=False)
    print(file1," and ",file2," is created and saved to csv")
    train_cech = pd.read_csv(file1, low_memory=False)
    test_cech = pd.read_csv(file2, low_memory=False)
    preprocessing(train_cech,name_of_file1)
    preprocessing(test_cech, name_of_file2)
    print("presiding to preprocessing")

# function that cals all required functions to create preprocessed files
def initiate():
    file1_exist = existing_files(file1)
    file2_exist = existing_files(file2)
    if file1_exist and file2_exist == True:
        print("file:", file1, " and ", file2, " already exist no new files crated")
        print("presiding to preprocessing...")
        train_cech = pd.read_csv(file1, low_memory=False)
        test_cech = pd.read_csv(file2, low_memory=False)
        preprocessing(train_cech, name_of_file1)
        preprocessing(test_cech, name_of_file2)
        print("if statement run")
    else:
        print("presiding to create_train_test")
        create_train_test()


# the preprocessing function to do the preprocessing
def preprocessing(dataframe,name_of_file):

    print("-----------------------------------Reading data ",name_of_file,"----------------------------------------------")
    # checkint the new saved dataset
    #print(test_Cech.head())
    print(dataframe.describe())
    print(dataframe.info())
    print(dataframe.head())
    print(dataframe.shape)
    #print(test_Cech.shape)
    print(dataframe.attack_cat.unique())
    print(dataframe.dtypes)
    print(dataframe.isnull().values.any())
    print(dataframe.isnull().any())
    # Counting NaN values in all columns
    nan_count = dataframe.isna().sum()
    print(nan_count)
    print(dataframe['ct_ftp_cmd'].unique(), ' in ', name_of_file)
    print(dataframe['is_ftp_login'].unique(), ' in ', name_of_file)

    print("-------------replacing nan values in", name_of_file, "-----------------------")
    # We don’t have “normal” values for “attack_cat,” so we must fill Null values with “normal”
    dataframe['attack_cat'] = dataframe.attack_cat.fillna(value='normal').apply(lambda x: x.strip().lower())
    # replacing nan values of ct_flw_http_mthd and is_ftp_login whit 0
    dataframe[["ct_flw_http_mthd", "is_ftp_login"]] = dataframe[["ct_flw_http_mthd", "is_ftp_login"]].fillna(0)
    dataframe['ct_flw_http_mthd'] = dataframe['ct_flw_http_mthd'].apply(np.int64)
    dataframe.attack_cat.value_counts()
    print(dataframe.attack_cat.unique(), "in ", name_of_file)
    nan_count = dataframe.isna().sum()
    print(nan_count,"in ", name_of_file)
    print(dataframe.dtypes)

    #############------------------################--------------#######
    # boolean column whit other number than bolen
    dataframe['is_ftp_login'] = np.where(dataframe['is_ftp_login'] > 1, 1, dataframe['is_ftp_login'])
    dataframe['is_ftp_login'] = dataframe['is_ftp_login'].apply(np.int64)
    ## re naming plural attack to singular
    dataframe['attack_cat'] = dataframe['attack_cat'].replace('backdoors', 'backdoor', )
    print(dataframe['attack_cat'].unique(), "in ", name_of_file)

    # In the research paper, it was mentioned that, this is a numerical feature and not categorical
    print("---")
    print("datatype of ct_ftp_cmd: ", dataframe['ct_ftp_cmd'].dtypes, " in ",name_of_file)
    print("unique valuess of ct_ftp_cmd: ", dataframe['ct_ftp_cmd'].unique()," in ",name_of_file)
    print("sum off empty string: ", (dataframe['ct_ftp_cmd'] == '').sum()," in ",name_of_file)
    print("summ of string whit spaces ' ': ", (dataframe['ct_ftp_cmd'] == ' ').sum()," in ",name_of_file)
    # identifying empty strings in the column that needs to be removed in order to convert to int64

    print("------------convert ct_ftp_cmd to numerical values by dropping some column in ",name_of_file,"-----------------")
    print("test")
    #Not the best solution to drop convert empty sting ' ' to nan adn than drop entire row
    # but works for now
    dataframe['ct_ftp_cmd'].replace(' ', np.nan, inplace=True)
    dataframe = dataframe.dropna()
    dataframe['ct_ftp_cmd'].apply(np.int64)
    print(name_of_file," hopefully converted", dataframe)
    print(dataframe['ct_ftp_cmd'].unique(), " in ", name_of_file)
    nan_count = dataframe.isna().sum()
    print(nan_count)
    print(dataframe['ct_ftp_cmd'].dtypes, " in ", name_of_file)

    ## re fixing the index after removed values
    #all_data.reset_index(drop=True)
    #df_col['Name'] = df_col['Name'].apply(lambda x: x.strip().replace(' ', '').lower())

    print("------------all preprocessing on ", name_of_file, " done printing results finished printing results------------------")
    print('shape of train_cech: ', dataframe.shape, " in ", name_of_file)
    print(dataframe.attack_cat.unique(), "in", name_of_file)
    print(dataframe.dtypes, " in ", name_of_file)
    nan_count = dataframe.isna().sum()
    print(nan_count," in ", name_of_file)
    print("---------")
    print("unique values of  ct_ftp_cmd", dataframe['ct_ftp_cmd'].unique(), " in ", name_of_file)
    print(dataframe['ct_ftp_cmd'].dtypes)
    print("unique values of is_ftp_login", dataframe['is_ftp_login'].unique(), " in ", name_of_file)
    print(name_of_file," ",dataframe.describe())
    print(name_of_file," ",dataframe.info())
    print(name_of_file + "_pp" + '.csv',)
    dataframe.to_csv("../Dataset/"+name_of_file + "_pp" + '.csv',index=False)


#train_cech = pd.read_csv(file1, low_memory=False)
#test_cech = pd.read_csv(file2, low_memory=False)

initiate()
#preprocessing(train_cech,name_of_file1)
#preprocessing(test_cech,name_of_file2)
print("alle done")

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