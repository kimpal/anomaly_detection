<h1> Anomaly Detection </h1>
<p> The raw network packets of the UNSW-NB 15 dataset was created by the IXIA PerfectStorm tool in the Cyber Range Lab of UNSW Canberra for generating a hybrid of real modern normal activities and synthetic contemporary attack behaviours. The tcpdump tool was utilised to capture 100 GB of the raw traffic (e.g., Pcap files). </p> 

<p> This dataset has nine types of attacks, namely, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode and Worms. The Argus, Bro-IDS tools are used and twelve algorithms are developed to generate totally 49 features with the class label. These features are described in UNSW-NB15_features.csv file.</p>


# Two ways of setting up an environment and installing all required packages:
1. without Docker
2. with Docker

## 1. Run normal and install all dependency:
1. Make sure to be in the directory where the requirements.txt file is
2. Installing all dependency:  `pip install --no-cache-dir --upgrade -r requerments.txt` or `pip install --upgrade pip && pip install --no-cache-dir --upgrade -r requerments.txt` or `pip install -r requerments.txt` 
3.  First run the pre-processing script in the dir /Pre-processing whit `python3 pre_prosessing_filename.py`
to generate the necessary csv files to run the models alw whit `python3 models_pythonfil_name.py`

## 2. Run with Docker:
1. The instruction assuming docker is already installed on Windows computer, docker desktop is recommended.
2. how to install docker desktop on windows [docker.com/desktop/install/windows-install](https://docs.docker.com/desktop/install/windows-install/)
3. On oter systems refer to: [docker.com/get-docker](https://docs.docker.com/get-docker/)
4. After install, follow the instruction below to run it and enter the shel where the anomaly-detection-main file is
### To start the container and enter it run:
1. `docker compose up --detach`
2. `docker exec -it anomaly-detection /bin/bash`
### To exit the docker container type: `exit`
### To shut down the docker container run: 
`docker compose down`

### to run python scrips: 
`python3 file_name.py`

# Running XGBoost FS Selection Method Step Bay Step
depending on Preprocessing for the UNSW-NB15 or TON-IoT and Bot-IoT binary / multi-class classification

## Example for UNSW-NB15 multiclass classification
1. make sure the UNSW-NB15 train and test dataset is in the directory Dataset
2. Run the python file: `Pre-processing_for_multiclass_validationsplit.py`
   and make sure that the files 'Dataset/train_1_pp3_multi.csv',
   'Dataset/val_pp3_multi.csv', and 'Dataset/test_pp3_multi.csv' getting created
3. Run the  `Feature_Importance_multiclass.py` File to get the importance score 
   for desired number of features to use in the last python file. 
4. `XBoost_Featureselecton_multiclass_models.py` open this file and
       Run it whit or whit out FS IF whit FS make sure to select the 
       corresponding feature important score corresponding to the number of desired features.
        The FS function is from line 54 to 80, the threshold values are given in line 65
5. in the same file `XBoost_Featureselecton_multiclass_models.py`
   the desired algorithm to fit and test is also selected in that python file.

### If wanted, step 4 code be skipped if it is desired to run al algorithms in one go
 Then run the file `XBoost_Feature_selection_multiclass_and_binary_models_automated_training.py` instead.
 Be aware that it could take several days to complete the run of all models depending on Hardware and dataset.
1. IN this file, the following needs to be done:
   - dataset needs to be selected
   - target values depending on dataset and binary / multiclass classification 
   - Feature importance score corresponding to the desired number of features discovered bay running the file form step 3 about
   - make sure the desired parameters to test are set for the different algorithm 
2. Start the code run and check that the code starts, then take a coffee brake...
