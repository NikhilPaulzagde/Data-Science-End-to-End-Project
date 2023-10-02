import os, sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    TRAIN_DATA_PATH :str = os.path.join("artifacts/Data Ingestion","train.csv")
    TEST_DATA_PATH :str = os.path.join("artifacts/Data Ingestion","test.csv")
    RAW_DATA_PATH :str = os.path.join("artifacts/Data Ingestion","raw.csv")

class DataIngestion:
    def __init__(self) :
        self.DataIngestionConfig=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion Started")
            data = pd.read_csv("notebook\data\Final_data.csv")
            logging.info("Data Reading Started")

            os.makedirs(os.path.dirname(self.DataIngestionConfig.RAW_DATA_PATH),exist_ok= True)
            data.to_csv(self.DataIngestionConfig.RAW_DATA_PATH, index=False , header= True)

            #Spliting Data 
            train_set, test_set = train_test_split(data, test_size=0.2, random_state= 6)
            logging.info("Data Has Been  Splitted")


            train_set.to_csv(self.DataIngestionConfig.TRAIN_DATA_PATH, index = False, header = True)
            test_set.to_csv(self.DataIngestionConfig.TEST_DATA_PATH, index = False, header = True)

            logging.info("Data Ingestion Completed")

            return(
                self.DataIngestionConfig.TRAIN_DATA_PATH,
                self.DataIngestionConfig.TEST_DATA_PATH
            )


        except Exception as e:
            logging.info("Error Found in initiate data ingestion")
            raise CustomException(e,sys)
