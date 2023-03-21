#Importing all the necessary libraries
import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception_handler import CustomException
from src.logger import logging
from src.utils import save_object

#Creating a Data transformation class:
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for the data transformation of the collected data from various sources
        '''


        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            #Creating a numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler(with_mean=False))
                ]
            )

            #Creating a categorical pipeline:
            categorical_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
                ]
            )

            #Logging information:
            logging.info('Missing values for both numerical and categorical data has been handled')
            logging.info('Numerical features standard scaling has been completed')
            logging.info('Categorical features encoding is completed')
            logging.info(f'Numerical Features: {numerical_features}')
            logging.info(f'Categorical Features: {categorical_features}')


            #Declaring the column transformer:
            preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_features),
                ('categorical_pipeline', categorical_pipeline, categorical_features)
            ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            #Reading the train and test data

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Train and test data reading completed..')

            #Obtaining the Preproccessing object

            preprocessing_ob = self.get_data_transformer_object()

            target_col_name = 'math_score'
            numerical_cols = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=[target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]

            #Processing/Transforming our data using the preprocessing object

            logging.info('Processing/Transforming our Train and Test data using the preprocessing object')


            input_feature_train_arr = preprocessing_ob.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_ob.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f'Saved the Preprocessing Object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                object = preprocessing_ob
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
