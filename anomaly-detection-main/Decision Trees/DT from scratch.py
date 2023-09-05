#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd


# In[59]:


test = pd.read_csv("../../Anomaly-Detection-main/Dataset/UNSW_NB15_testing-set.csv", sep=',', header=0)
train = pd.read_csv("..//../Anomaly-Detection-main/Dataset/UNSW_NB15_training-set.csv", sep=',', header=0)


# In[60]:


X_train = train.drop(['label'], axis=1)
X_test = test.drop(['label'], axis=1)
y_train = train.loc[:, ['label']]
y_test = test.loc[:, ['label']]


# In[61]:


def accuracy_metric(actual, predictec):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predict[i]:
            correct += 1
        return correct/float(len(actual))*100


# In[62]:


#Splitting the dataset

def test_split(index, value, train):
    left, right = list(), list()
    for row in train:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
        return left, right


# In[63]:


#Gini index

def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val)/ size
            score+= p * p
        gini += (1.0 - score) * (size/n_instances)
    return gini


# In[64]:


#selecting the best split for the dataset

def get_split(train):
    class_values = list(set(row[-1] for row in train))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(train[0])-1):
        groups in test_split(index, row[index], train)
        gini = gini_index(groups, class_values)
        if gini < b_score:
            b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return{'index':b_index, 'value':b_value, 'groups':b_groups}


# In[65]:


#create a terminal node
def terminal(group):
    outcome = [row[-1] for row in group]
    return max(set(outcome), key=outcome.count)


# In[66]:


#create child node splits for a node or make it terminal
def split(node, max_depth, min_size, depth):
    left, right = node('groups')
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = terminal(left), terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)


# In[67]:


#Tree building \o/
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# In[68]:


#Making the prediction with the decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right', row])
        else:
            return node['right']


# In[69]:


def dt(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    prediction = list()
    for row in test:
        prediction = predict(tree, row)
        prediction.append(prediction)
    return(prediction)


# In[70]:


def evaluate(algorithm, train, test, *args):
    score = list()
    predicted = algorithm(train, test, *args)
    actual = test.iterrows()
    accuracy = accuracy_metric(actual, predicted)
    return score


# In[71]:


scores = evaluate(dt, X_train, X_test, 5, 10)
print(score)

