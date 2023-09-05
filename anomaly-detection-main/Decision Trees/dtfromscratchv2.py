#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np


# In[55]:


train = pd.read_csv("../../Anomaly-Detection-main/Dataset/UNSW_NB15_training-set.csv", sep=',', header=0)
test = pd.read_csv("../../Anomaly-Detection-main/Dataset/UNSW_NB15_testing-set.csv", sep=',', header=0)



# In[56]:


#calculating the entropy. Taking the label from the dataset as input.

def calculate_entropy(label):
    classes, class_counts= np.unique(label, return_counts = True)
    entropy_value = np.sum([(-class_counts[i]/np.sum(class_counts))*np.log2(class_counts[i]/np.sum(class_counts))for i in range(len(classes))])
    return entropy_value
       


# In[57]:


#calculate the information gain
def calculate_information_gain(dataset, feature, label):
    dataset_entropy = calculate_entropy(dataset[label])
    values,feat_counts= np.unique(dataset[feature],return_counts=True)

    #calculate the weighted feature entropy
    weighted_feature_entropy = np.sum([(feat_counts[i]/np.sum(feat_counts))*calculate_entropy(dataset.where(dataset[feature]==values[i]).dropna()[label])for i in range(len(values))])
    feature_info_gain = dataset_entropy = weighted_feature_entropy
    return feature_info_gain


# In[58]:


#creating the decision tree
def create_decision_tree(dataset,df,features,label,parent):
    datam = np.unique(df[label],return_counts=True)
    unique_data = np.unique(dataset[label])

    if len(unique_data) <= 1:
        return unique_data[0]
    elif len(dataset) == 0:
        return unique_data[np.argmax(datam[1])]
    elif len(features) == 0:
        return parent
    else:
        parent = unique_data[np.argmax(datam[1])]

        item_values = [calculate_information_gain(dataset, feature, label) for feature in features]

        for value in np.unique(dataset[optimum_feature]):
            min_data = dataset.where(dataset[optimum_feature]==value).dropna()

            min_tree = create_decision_tree(min_data,df,features,label,parent)

            decision_tree[optimum_feature][value] = min_tree
        return(decision_tree)


# In[59]:


features = train.columns[:-1]
label = 'label'
parent=None


# In[60]:


dt = create_decision_tree(train,train, features,label,parent)


# In[ ]:


test_data = pd.series(test)
pred = predict_attack(test_data, dt)
pred

