# Correlation Method

This project utilized ```python version 3.12.4``` as the coding language. To install the required packages, run the ```req.py```file. 


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
<br>Line 36 to 41: 
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

The models are in folder *Model_Code*.
Each model contains a code block that looks like this:
```
start = 0
start2 = 0

data = 2

end = len(task1) #-1
end2= len(task1) #-1
```
data decided the dataset used ranging from 1-3, where 1 is UNSW-NB15, 2 is ToN and 3 is BoT.

start decide the type of classification to start with ranges from 0-1, where 0 is binary and 1 is multi class. (Default: ```start = 0```)
<br>
start2 decide the feature to start with ranges from 0-1, where 0 is reduced and 1 is full feature. (Default: ```start2 = 0```)

end decides if you want to exclusivly run binary. If thats the case put -1 at the end, if not do nothing. (Default: ``` end = len(task1) ```)
<br>
end2 decides if you want to exclusivly run reduced feature. If thats the case put -1 at the end, if not do nothing. (Default: ``` end = len(task1) ```)
