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
                'K neighbors regressor' : KNeighborsRegressor()
            }


            params = {
                'Random Forest' : {
                'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'max_features' : ['sqrt', 'log2', None],
                'n_estimators' : [8, 16, 32, 64, 256],
                },

                'Decision Tree' : {
                'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'splitter' : ['best', 'random'],
                'max_features' : ['sqrt', 'log2']
                },

                'Ada Boost Regressor' : {
                'learning_rate' : [.1, .01, .5, .001],
                'loss' : ['linear', 'square', 'exponential'],
                'n_estimators' : [8, 16, 32, 64, 128, 256]
                },
                
                'Gradient Boosting' : {
                'loss' : ['squared_error', 'hubert', 'absolute_error', 'quantile'],
                'learning_rate' : [.1, .01, .5, .001],
                'criterion' : ['squared_error', 'friedman_mse'],
                'max_features' : ['sqrt', 'log2', 'auto'],
                'n_estimators' : [8, 16, 32, 64, 128, 256]
                },

                'Linear Regression' : {
                
                },

                'Cat Boost Regression' : {
                'depth' : [6, 8, 10],
                'learning_rate' : [.1, .01, .05, .001],
                'iterations' : [30, 50, 100]
                },

                'XGBRegressor' : {
                'learning_rate' : [.1, .01, .5, .001],
                'n_estimators' : [8, 16, 32, 64, 128, 256]
                },

                'K neighbors regressor' : {
                'n_neighbors' : [5, 10, 15, 20],
                'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']
                },

            }


            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, 
                                               X_test=X_test, y_test=y_test, models=models, params=params)
            

            #Getting the model which performed the best
            best_model_score = max(sorted(model_report.values()))

            #The model's name:
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]


            if best_model_score < 0.6: #setting a threshold for the model_score, if prediction less than 60% accuracy
                raise CustomException("No best models")
            
            logging.info('Best model on Training and testing dataset found')
            print(f"Best Model: {best_model_name}")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )

            #Predictions made by the best model in the set:
            predictions = best_model.predict(X_test)

            r2_val = r2_score(y_test, predictions)
            logging.info('R2 Score: ', r2_val)
            return r2_val

        except Exception as e:
            raise CustomException(e, sys)

