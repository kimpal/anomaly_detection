# Correlation Method

* The folder **Cor_Code** contains the code for the correlation method.
* The folder **Dataset** within Cor_code should contain the datasets from UNSW-NB15, ToN_IoT and BoT-IoT.
### The datasets should have these names:
  #### BoT-IoT
  - IoT_Botnet_Test
  - IoT_Botnet_Train
  #### TON_IoT  
  - TON_train_test
  #### UNSW-NB15
  - UNSW-NB15_Train
  - UNSW-NB15_Test

The folder **Used** within the Dataset folder is where the preprocessed csv files will be stored.

To obtain the preproccesed datasets, you need to run ```Own_Preprocessing_v3.py```.
Line 36 to 41: 
```
d = 3  
t = 2

score = False
save = False
Time_complex = False
```
controls the behaviour of the preproccessed output. 
* *d* controls the dataset, where 1 is UNSW, 2 is ToN, and 3 is BoT
* *t* controls if its binary or multiclass, where 1 is binary and 2 is multiclass
* *score* controls if the dataset should tested with crossvalidation
* *save* controls if the produced preproccessed dataset should be saved as a csv file
* *Time_complex* controls if you want to measure the timecomplexity of the preproccessed datasets
