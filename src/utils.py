'''this file contains utility functions, that will be used throughout the project'''

import os
import sys

import numpy as np
import pandas as pd
import dill 
from sklearn.metrics import r2_score

from src.exception_handler import CustomException


def save_object(file_path, object):
    try:
        
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(object, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train) #Training model[fitting the models to the training dataset]

            y_train_preds = model.predict(X_train)

            y_test_preds = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_preds)

            test_model_score = r2_score(y_test, y_test_preds)

            report[list(models.keys())[i]] = test_model_score

        return report 
    
    except Exception as e:
        raise CustomException(e, sys)
        