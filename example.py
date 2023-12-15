import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging 

logging.basicConfig(level = logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__=="__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)


    # Read teh wine-quality csv file from thr url
    csv_url = ""
    try:
        data =  pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download csv")

    # Split the data into trainin g and test test sets
    train, test = train_test_split(data)


    # The predicted column is quality which is a scaler from [3,9]
    train_x = train.drop(["quality"], axis = 1)
    test_x = test.drop(["quality"], axis = 1)
    train_y = train["quality"]
    test_y = test["quality"]


    alpha = float(sys.argv[1]) if len(sys.argv)> 1 else 0.5
    
