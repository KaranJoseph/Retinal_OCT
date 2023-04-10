# Retinal_OCT
Federated Learning with Differential Privacy + MLflow &amp; Optuna


CNN Framework with extensive hyperparamter optimization using optuna and model tracking with mlflow. Best performing model is use as baseline for Federated Learning and Federated Learning with Differential Privacy. 

<u>Instructions for use:</u>
- create your virtual env -> activate
- pip install -r requirements.txt
    - source folder is the brain
    - data folder has a sample dataset for less intensive experiments
    - helpers: any helper function to prepare the dataset
- src:
    - 3 folders centralized, federated, and federated_dp holds the codes corresponding to the 3 approaches
    - models - final model parameters
- run **predict.py** to evaluate on test data
