'''Modular code for data ingestion(reading data from sources)'''

#Importing all necessary libraries
import os
import sys
from src.exception_handler import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv('notebooks\dataset\student.csv')
            logging.info('Read the dataset successfully from the source!')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) #Adding directory for the Train Data

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) #Saving our dataframe to csv format in the specified path


            logging.info('Train Test Split Initiated...')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion is completed...')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    ingest = DataIngestion()
    ingest.initiate_data_ingestion()

