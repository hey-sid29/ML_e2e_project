#Importing all the necessary libraries
import os
import sys
from dataclasses import dataclass


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, 
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.exception_handler import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting our data in training and testing samples')

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            models = {
                'Random Forest' : RandomForestRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Ada Boost Regressor' : AdaBoostRegressor(),
                'Gradient Boosting' : GradientBoostingRegressor(),
                'Linear Regression' : LinearRegression(),
                'Cat Boost Regression' : CatBoostRegressor(verbose=False),
                'XGBRegressor' : XGBRegressor(),
                'K nearest neighbors regressor' : KNeighborsRegressor()
            }


            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, 
                                               X_test=X_test, y_test=y_test, models=models)
            

            #Getting the model which performed the best
            best_model_score = max(sorted(model_report.values()))

            #The model's name:
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]


            if best_model_score < 0.6: #setting a threshold for the model_score, if prediction is 60% accurate
                raise CustomException("No best models")
            
            logging.info('Best model on Training and testing dataset found')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )

            #Predictions made by the best model in the set:
            predictions = best_model.predict(X_test)

            r2_val = r2_score(y_test, predictions)
            return r2_val

        except Exception as e:
            raise CustomException(e, sys)

