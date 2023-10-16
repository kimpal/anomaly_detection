

# imports
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

train_dataset = "./train_pp3_multi.csv"
test_dataset =  "./test_pp3_multi.csv"

train = pd.read_csv(train_dataset)
test = pd.read_csv(test_dataset)
print('dataset in shape of train: ', train.shape)
print('dataset in shape of tes: ', test.shape)

# split data into X an Y
X_train_multi, y_train_multi = train.drop(['attack_cat'], axis=1), train['attack_cat']
X_test_multi, y_test_multi = test.drop(['attack_cat'], axis=1), test['attack_cat']

#max_depth 20 19 18 17 16 15 14
# Create the parameter grid based on the results of random search
print("grid search start...")
param_grid = {
    'bootstrap': [True],
    'max_depth': [9,10,11,12,13,14,15,16,17,18,19],
    'max_features':["sqrt", "log2", None],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'min_samples_split': [2, 4, 6, 8, 10,11,12],
    'n_estimators': [10,11,12,13,14,15,16,17,18,19,20,21],
    'criterion':["gini", "entropy", "log_loss"],
    'min_weight_fraction_leaf':[0.0],
    'max_leaf_nodes':[None],
    'min_impurity_decrease':[0.0],
    'random_state':[None],
    'verbose':[0],
    'warm_start':[False],
    'class_weight':["balanced", "balanced_subsample", None],
    'ccp_alpha':[0.0],
    'max_samples':[None]
}

# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, verbose = 2)#, error_score='raise' , n_jobs = -1
# Fit the grid search to the data
grid_search.fit(X_train_multi, y_train_multi)

print(grid_search.best_params_)
print(grid_search.best_estimator_)
print(grid_search.best_score_)
print("--------------------------------------------------------------------------")
print("Finish")
