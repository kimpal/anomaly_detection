<h1> Anomaly Detection </h1>
<p> The raw network packets of the UNSW-NB 15 dataset was created by the IXIA PerfectStorm tool in the Cyber Range Lab of UNSW Canberra for generating a hybrid of real modern normal activities and synthetic contemporary attack behaviours. The tcpdump tool was utilised to capture 100 GB of the raw traffic (e.g., Pcap files). </p> 

<p> This dataset has nine types of attacks, namely, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode and Worms. The Argus, Bro-IDS tools are used and twelve algorithms are developed to generate totally 49 features with the class label. These features are described in UNSW-NB15_features.csv file.</p>

# two ways of setting up a envierment and instal requered pacages

## to runn whit out Docker pip install --no-cache-dir --upgrade -r requerments.txt
1. install all dependency: # not tested
`pip install --upgrade pip && pip install --no-cache-dir --upgrade -r requerments.txt`

1. installing all dependency whit: 
`pip install --no-cache-dir --upgrade -r requermnt.txt` # not testted
2. thna first runn the pre-prosessing script in the dir /Pre-prosessing
to generate the nesesary csv files to runn the models


## Run instructions To runn it whit docker 
1. Te instruction asuming docker is already installed on windows computer docker desktop is recomanded
2. how to install docker desktop on windows https://docs.docker.com/desktop/install/windows-install/
3. On oter systems refer to: https://docs.docker.com/get-docker/
4. After install fowo the instruction below to runn it and enter the shel wher the amomaly-detection-main file is
# To start the contianeer and enter it runn:
1. `docker compose up --detach`
2. `docker exec -it anomaly-detection /bin/bash`
# To exit the docker dontainer type: exit
# to sut down the docker container runn: 
`docker compose down`

# to runn python scrips: python3 file_name
