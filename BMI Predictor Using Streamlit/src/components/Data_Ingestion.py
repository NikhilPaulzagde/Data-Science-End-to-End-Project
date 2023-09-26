import os , sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("Artifacts/Data Ingestion","Train.csv")
    test_data_path: str = os.path.join("Artifacts/Data Ingestion","Test.csv")
    raw_data_path: str = os.path.join("Artifacts/Data Ingestion","Raw.csv")

class DataIngestion:
    def __init__(self) :
        self.data_ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        try:
            data = pd.read_csv("notebook\data\Final_Data.csv")

            logging.info("Data Has Been Reading as DataFrame")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok= True)

            data.to_csv(self.data_ingestion_config.raw_data_path, index= False, header=True)

            # Spliting to Test and Train Data
            logging.info("Data Has Splited")
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=5)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index = False, header = True)

            test_set.to_csv(self.data_ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion Completed")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
            


        except Exception as e:
            logging.info("Error Found In Data Ingestion")
            raise CustomException (e,sys)


