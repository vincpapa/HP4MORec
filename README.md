# Sensitivity of Conventional Recommender Systems to Hyperparameter Tuning when Considering Beyond Accuracy Objectives

This repository contains the codes of the paper ''Tuning Conventional Recommender Systems for Beyond Accuracy
Objectives''. In this paper, we propose a framework to analyze to what extent accuracy-oriented recommender systems need precise hyperparameter tuning to achieve Pareto optimal solutions when dealing with multiple (conflicting) objectives.

## Directory Structure

```plaintext
HP4MORec/
├── conf_files                  # The configuration files used to train the models using the Elliot framework.  
├── README.md                   # This file.
├── requirements.txt            # Python dependencies.
├── requirements.txt            # File to run for the content exposure scenario.
├── data/                       # File to run for the content delivery scenario.
└── src/                        # Source code for the framework.
```

## Environment Setup
Follow these steps to set up your environment and utilize the framework.
After creating your project directory `<your working directory>`, run the following commands.
```
cd <your working directory>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the Paper's Results
We have exploited the framework Elliot to train the models. Please, refer to the official documentation for training the
models. In this repository, we provide the configuration files used to guide the models' training in the `conf_files` directory
and the computed performance of each model in the `data` directory.

Follow these instructions to compute the results of the paper.
### 1. Content Exposure Scenario
Run the following command.
```
python3 content_exposure.py
```
### 2. Content Delivery Scenario
Run the following command.
```
python3 content_delivery.py
```

## Support Material

### Amazon Books Dataset
#### 1. Content exposure scenario
**GRAPH-BASED RECOMMENDER SYSTEMS**
<img src="img/books-graph-exposure.png">
**FACTORIZATION-BASED RECOMMENDER SYSTEMS**
<img src="img/books-fact-exposure.png">
**NEIGHBORHOOD-BASED RECOMMENDER SYSTEMS**
<img src="img/books-neigh-exposure.png">
#### 2. Content delivery scenario
**GRAPH-BASED RECOMMENDER SYSTEMS**
<img src="img/books-graph-delivery.png">
**FACTORIZATION-BASED RECOMMENDER SYSTEMS**
<img src="img/books-fact-delivery.png">
**NEIGHBORHOOD-BASED RECOMMENDER SYSTEMS**
<img src="img/books-neigh-delivery.png">

### Movielens1M Dataset
#### 1. Content exposure scenario
**GRAPH-BASED RECOMMENDER SYSTEMS**
<img src="img/ml1m-graph-exposure.png">
**FACTORIZATION-BASED RECOMMENDER SYSTEMS**
<img src="img/ml1m-fact-exposure.png">
**NEIGHBORHOOD-BASED RECOMMENDER SYSTEMS**
<img src="img/ml1m-neigh-exposure.png">
#### 2. Content delivery scenario
**GRAPH-BASED RECOMMENDER SYSTEMS**
<img src="img/ml1m-graph-delivery.png">
**FACTORIZATION-BASED RECOMMENDER SYSTEMS**
<img src="img/ml1m-fact-delivery.png">
**NEIGHBORHOOD-BASED RECOMMENDER SYSTEMS**
<img src="img/ml1m-neigh-delivery.png">

### Amazon Music Dataset
#### 1. Content exposure scenario
**GRAPH-BASED RECOMMENDER SYSTEMS**
<img src="img/music-graph-exposure.png">
**FACTORIZATION-BASED RECOMMENDER SYSTEMS**
<img src="img/music-fact-exposure.png">
**NEIGHBORHOOD-BASED RECOMMENDER SYSTEMS**
<img src="img/music-neigh-exposure.png">
#### 2. Content delivery scenario
**GRAPH-BASED RECOMMENDER SYSTEMS**
<img src="img/music-graph-delivery.png">
**FACTORIZATION-BASED RECOMMENDER SYSTEMS**
<img src="img/music-fact-delivery.png">
**NEIGHBORHOOD-BASED RECOMMENDER SYSTEMS**
<img src="img/music-neigh-delivery.png">