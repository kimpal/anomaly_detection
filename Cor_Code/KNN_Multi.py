#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbor

# In[1]:

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Dataset loading
import sys
sys.path.append("..")
from sklearn.preprocessing import MinMaxScaler

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

# In[2]:
    
task1=["Binary", "Multi"]

start = 1
start2 = 0

data = 2

end = len(task1) #-1
end2= len(task1) -1

if data ==1:
   Dataset = "UNSW_NB15"
   task2=["21FE", "42FE"]
   task3=["22FE", "42FE"]
if data == 2:
   Dataset = "TON_Train_Test"
   task2=["17FE", "41FE"] # 18 43 
   task3=["17FE", "41FE"] # 17 19 43
if data == 3:
   Dataset = "IoT_Botnet"
   task2=["9FE", "13FE"]  #13 17   --10    ----- 9 -- 13
   task3=["12FE", "16FE"]  #12 17           ------9 -- 13

for x in range(start, end):
   for y in range(start2, end2):
        
        TYPE = task1[x] #Multi, Binary
        FEATURES = task2[y]
        print(Dataset)
        print(task1[x])
           
        Min = 194
        Max = 194 
        if x == 1:
            FEATURES = task3[y]
            print(task3[y])
            #Min = 9
            #
            Max = 194
        else:
            print(task2[y])
          #UNSW_NB15: 42FE, 22FE Multi, 21FE Binary    # RUN UNSW ALL, TON ALL and BOT IOT all FOR RUNTIME
                           #Bot: 17FE, 12FE Multi, 13FE Binary
                           #Ton: 43FE, 19FE Multi, 18FE Binary
        if TYPE == "Multi":
           av = "weighted"
        else:
           av = "binary"
        
        
        if Dataset == "UNSW_NB15" and FEATURES == "22FE" and TYPE == "Multi": 
          x_train = pd.read_csv(f'../Dataset/Used/{Dataset}_xtrain_{FEATURES}_{TYPE}_v1.csv')
          y_train = pd.read_csv(f'../Dataset/Used/{Dataset}_ytrain_{FEATURES}_{TYPE}_v1.csv')
          
          x_test = pd.read_csv(f'../Dataset/Used/{Dataset}_xtest_{FEATURES}_{TYPE}_v3.csv')
          y_test = pd.read_csv(f'../Dataset/Used/{Dataset}_ytest_{FEATURES}_{TYPE}_v3.csv')
        else:
          x_train = pd.read_csv(f'../Dataset/Used/{Dataset}_xtrain_{FEATURES}_{TYPE}.csv')
          y_train = pd.read_csv(f'../Dataset/Used/{Dataset}_ytrain_{FEATURES}_{TYPE}.csv')
        
          x_test = pd.read_csv(f'../Dataset/Used/{Dataset}_xtest_{FEATURES}_{TYPE}.csv')
          y_test = pd.read_csv(f'../Dataset/Used/{Dataset}_ytest_{FEATURES}_{TYPE}.csv')
        
        x_train = scaler1.fit_transform(x_train)
        x_test = scaler2.fit_transform(x_test)
        
       
        
        Path = f'{Dataset}_K_{Max}_{TYPE}_{FEATURES}.csv'
        

        error_test = []
        K_value = []
        
        train_acc = []
        test_acc = []
        precision_train = []
        precision_test = []
        F1_train = []
        F1_test = []
        recall_train = []
        recall_test = []
        
        import time
        
        timer = []
        
         
            
        # Calculating error for K values between 1 and 200 and appending scores to lists
        
        print(f"Running KNN (1-{Max})")

        
        for i in range(Min, Max+1):
            
            knn = KNeighborsClassifier(n_neighbors=i, n_jobs=1)
            
            start = time.time()
            knn.fit(x_train, np.ravel(y_train)) 
            pred_i_test = knn.predict(x_test)
            pred_i_train = knn.predict(x_train)
            end = time.time()
            timer.append(end-start)
            # Appending values to list
            #error_test.append(np.mean(pred_i_test != y_test))
            K_value.append(i)
            train_acc.append(metrics.accuracy_score(y_train, pred_i_train))
            test_acc.append(metrics.accuracy_score(y_test, pred_i_test))
            
            precision_train.append(metrics.precision_score(y_train, pred_i_train, average=av, zero_division=0.0))
            precision_test.append(metrics.precision_score(y_test, pred_i_test, average=av, zero_division=0.0))
            
            F1_train.append(metrics.f1_score(y_train, pred_i_train, average=av, zero_division=0.0))
            F1_test.append(metrics.f1_score(y_test, pred_i_test, average=av, zero_division=0.0))
            
            recall_train.append(metrics.recall_score(y_train, pred_i_train, average=av, zero_division=0.0))
            recall_test.append(metrics.recall_score(y_test, pred_i_test, average=av, zero_division=0.0))
            
            # dictionary of lists 
            dict = {
                    'K': K_value,
                    'train_acc': train_acc, 
                    'test_acc': test_acc, 
                    'precision_train': precision_train,
            	'precision_test': precision_test,
                    'F1_train': F1_train,
            	'F1_test': F1_test,
                    'recall_train': recall_train,
            	'recall_test': recall_test,
            	'Runtime': timer,
                    #'error': error_test
                    }
            print("Storing as csv file....")       
            df = pd.DataFrame(dict, index=K_value)
            df.set_index("K", inplace = True)
            # EXPORT AS CSV when done.
            df.to_csv(Path) 
            print("Opening csv file....")
            new_dataframe = pd.read_csv(Path)
            gg = new_dataframe.sort_values(by=['test_acc'], ascending = False)
            print(gg.to_string())
# In[1]:        

    
"""
 42 Multi
 
 Opening csv file....
      K  train_acc  test_acc  precision_train  precision_test  F1_train   F1_test  recall_train  recall_test     Runtime
 10  11   0.802391  0.708935         0.803427        0.786767  0.798588  0.733531      0.802391     0.708935  107.012869
 8    9   0.803520  0.706675         0.806363        0.786625  0.801725  0.733106      0.803520     0.706675   97.359788
 9   10   0.802602  0.704368         0.805603        0.787954  0.800266  0.730718      0.802602     0.704368  104.592411
 6    7   0.805938  0.702558         0.810044        0.786237  0.805897  0.731944      0.805938     0.702558  113.557353
 4    5   0.809081  0.700104         0.819048        0.786276  0.812416  0.731542      0.809081     0.700104  101.789483
 7    8   0.804421  0.699862         0.809322        0.787806  0.803924  0.728656      0.804421     0.699862   98.348761
 5    6   0.807484  0.694821         0.816534        0.788605  0.809735  0.727462      0.807484     0.694821  103.051854
 2    3   0.813489  0.689586         0.840676        0.784102  0.824720  0.725974      0.813489     0.689586  102.376140
 0    1   0.867818  0.687011         0.888382        0.778623  0.873763  0.723141      0.867818     0.687011  104.827623
 3    4   0.808242  0.686295         0.829117        0.789899  0.816469  0.723336      0.808242     0.686295   97.486365
 1    2   0.814168  0.664468         0.865664        0.789807  0.830862  0.709805      0.814168     0.664468  109.535348
 KNN run is over
 
 
"""
 
"""
 22 Multi
 
      K  train_acc  test_acc  precision_train  precision_test  F1_train   F1_test  recall_train  recall_test    Runtime
 10  11   0.903057  0.755544         0.915181        0.821439  0.900952  0.782740      0.903057     0.755544  20.901094
 8    9   0.905073  0.750934         0.919061        0.820453  0.903524  0.779152      0.905073     0.750934  20.933672
 11  12   0.901960  0.749668         0.914038        0.814403  0.899998  0.775677      0.901960     0.749668  21.263407
 9   10   0.904302  0.748047         0.918528        0.817584  0.903241  0.776372      0.904302     0.748047  21.092918
 6    7   0.907197  0.747793         0.922965        0.821122  0.906448  0.777340      0.907197     0.747793  21.197678
 4    5   0.911258  0.742183         0.929014        0.820601  0.911612  0.772350      0.911258     0.742183  21.843228
 7    8   0.906269  0.740220         0.921930        0.815529  0.905590  0.770071      0.906269     0.740220  21.030595
 5    6   0.908650  0.740182         0.926344        0.821902  0.908927  0.772668      0.908650     0.740182  21.058786
 3    4   0.913185  0.606677         0.934996        0.805505  0.915157  0.663377      0.913185     0.606677  22.335876
 2    3   0.916525  0.601370         0.940421        0.789278  0.918444  0.651866      0.916525     0.601370  21.285694
 0    1   0.933431  0.577258         0.964549        0.794934  0.938014  0.631021      0.933431     0.577258  20.991251
 1    2   0.912968  0.568557         0.942734        0.804165  0.918859  0.628084      0.912968     0.568557  21.276633
 KNN run is over
 
 
"""
 
"""
 42 Binary
 
 Opening csv file....
      K  train_acc  test_acc  precision_train  precision_test  F1_train   F1_test  recall_train  recall_test    Runtime
 1    2   0.965770  0.867609         0.999621        0.844913  0.974214  0.885559      0.950067     0.930314  75.983095
 3    4   0.958863  0.859362         0.981558        0.823298  0.969406  0.881281      0.957550     0.948050  70.466863
 5    6   0.955458  0.852172         0.970909        0.809736  0.967152  0.876890      0.963424     0.956190  70.729149
 0    1   0.997120  0.850423         0.998975        0.807966  0.997882  0.875527      0.996791     0.955418  75.920877
 7    8   0.952875  0.848042         0.964863        0.801780  0.965400  0.874527      0.965938     0.961793  70.925297
 2    3   0.968587  0.846900         0.973566        0.798620  0.977005  0.874112      0.980468     0.965367  71.142229
 9   10   0.951837  0.845528         0.960539        0.796610  0.964774  0.873213      0.969047     0.966117  70.760689
 4    5   0.959234  0.843742         0.963236        0.793410  0.970271  0.872192      0.977409     0.968345  70.704432
 6    7   0.955527  0.841107         0.958033        0.788916  0.967657  0.870660      0.977476     0.971301  70.745051
 8    9   0.953240  0.839552         0.954412        0.786088  0.966068  0.869816      0.978013     0.973507  70.775193
 10  11   0.951443  0.838580         0.951460        0.784333  0.964830  0.869293      0.978582     0.974896  70.954362
 KNN run is over
"""
 
"""
 21 Binary
 
 Opening csv file....
      K  train_acc  test_acc  precision_train  precision_test  F1_train   F1_test  recall_train  recall_test    Runtime
 5    6   0.955572  0.865909         0.977281        0.832919  0.967020  0.885990      0.956972     0.946285  46.922242
 3    4   0.958367  0.864694         0.984971        0.836231  0.968917  0.884173      0.953377     0.937947  47.144887
 7    8   0.953810  0.863783         0.973483        0.828320  0.965800  0.884725      0.958237     0.949374  46.970160
 1    2   0.964270  0.862459         0.999100        0.843668  0.973068  0.880561      0.948358     0.920829  48.939484
 9   10   0.952983  0.861597         0.970268        0.823919  0.965283  0.883388      0.960349     0.952109  50.538746
 11  12   0.952145  0.859824         0.967363        0.820276  0.964749  0.882337      0.962150     0.954557  48.981957
 6    7   0.956342  0.857115         0.967901        0.814342  0.967929  0.880842      0.967957     0.959168  46.914468
 8    9   0.954181  0.855451         0.965295        0.812095  0.966378  0.879655      0.967463     0.959477  47.267895
 4    5   0.959182  0.854698         0.971742        0.812555  0.969959  0.878810      0.968184     0.956830  46.833300
 10  11   0.953759  0.853799         0.963784        0.809462  0.966112  0.878570      0.968452     0.960580  51.255749
 2    3   0.965194  0.850593         0.978694        0.809354  0.974316  0.875394      0.969977     0.953168  46.903343
 0    1   0.987561  0.849524         0.998536        0.812073  0.990791  0.873727      0.983166     0.945513  48.413350
 KNN run is over
"""