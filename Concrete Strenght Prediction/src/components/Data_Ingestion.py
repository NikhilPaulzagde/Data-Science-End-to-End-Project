import os, sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("Artifacts/Data_Ingestion","train.csv")
    test_data_path = os.path.join("Artifacts/Data_Ingestion","test.csv")
    raw_data_path = os.path.join("Artifacts/Data_Ingestion","raw.csv")
    

class DataIngestion:
    def __init__(self) :
        self.data_ingestion_config = DataIngestionConfig()

# 
    def initiate_data_ingestion(self):
        try:
            logging.info("Data Reading using Pandas")
            data = pd.read_csv('notebook\data\concrete_data.csv')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.data_ingestion_config.raw_data_path , index=False)
            logging.info("Raw Data is Saved")

            Train_set,Test_set=train_test_split(data, test_size = 0.25, random_state = 6)
            logging.info("Data Has Splitted Into Train And Test")
        
            Train_set.to_csv(self.data_ingestion_config.train_data_path, index = False, header = True)
            Test_set.to_csv(self.data_ingestion_config.test_data_path, index = False, header = True)
            
            logging.info("Data Ingestion is Completed ")

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Data ingestion is Unsucessfull")
            raise CustomException(e,sys)