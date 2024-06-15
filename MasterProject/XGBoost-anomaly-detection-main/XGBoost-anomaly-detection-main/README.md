# Rep for the XGBoost FS Method

This branch contains only the code
used in the XGBoost FS Method and its preprocessing and algorithms for model fit and testing
If the entire code form previous method is desired it is referred to the Main branch:
[anomaly-detection-main](https://github.com/kimpal/anomaly_detection/tree/XGBoost_FS_Method_only/anomaly-detection-main#readme) 

# Running XGBoost FS Selection Method Step Bay Step
depending on Preprocessing for the UNSW-NB15 or TON-IoT and Bot-IoT binary / multi-class classification

## First navigate to work dir and install required packages

1. navigate to: anomaly-detection-main

2. upgrade pip: `python.exe -m pip install --upgrade pip` 

3. install all required packages with: `pip install -r requirements.txt`

## Example for UNSW-NB15 multiclass classification
1. make sure the UNSW-NB15 train and test dataset is in the directory Dataset
2. Run the python file: `python pre-processing\Pre-processing_for_multiclass_validationsplit.py`
   and make sure that the files 'Dataset/train_1_pp3_multi.csv',
   'Dataset/val_pp3_multi.csv', and 'Dataset/test_pp3_multi.csv' getting created
3. Run the  `python anomaly-detection-main\pre-processing\Feature_Importance_multiclass.py` File to get the importance score 
   for desired number of features to use in the last python file. 
4. `python XboostFeatureselection_multipel_binary_clasifyer_and_multi_classifyer\XBoost_Featureselecton_multiclass_models.py` open this file and
       Run it whit or whit out FS IF whit FS make sure to select the 
       corresponding feature important score corresponding to the number of desired features.
        The FS function is from line 54 to 80, the threshold values are given in line 65
5. in the same file `XBoost_Featureselecton_multiclass_models.py`
   the desired algorithm to fit and test is also selected in that python file.

### If wanted, step 4 code be skipped if it is desired to run al algorithms in one go
 Then run the file `python boostFeatureselection_multipel_binary_clasifyer_and_multi_classifyer\XBoost_FS_automated_run_binary_multi_class_training.py`instead.
 Be aware that it could take several days to complete the run of all models depending on Hardware and dataset.
1. IN this file, the following needs to be done:
   - dataset needs to be selected
   - target values depending on dataset and binary / multiclass classification 
   - Feature importance score corresponding to the desired number of features discovered bay running the file form step 3 above
   - make sure the desired parameters to test are set for the different algorithm 
2. Start the code run and check that the code starts, then take a coffee brake...
