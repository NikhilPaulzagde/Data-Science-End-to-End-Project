import os , sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionconfig:
    train_data_path=os.path.join("artifacts","train.csv")
    test_data_path=os.path.join("artifacts","test.csv")
    raw_data_path=os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestionconfig=DataIngestionconfig()

    def initiate_data_ingestion(self):
        try:
            data=pd.read_csv("notebook/data/data.csv")
            logging.info("Data has been read successfully")


            os.makedirs(os.path.dirname(self.ingestionconfig.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestionconfig.raw_data_path, index=False)
            logging.info("Raw Data Has been Saved")

            train_set,test_set=train_test_split(data, test_size=0.20,random_state=2)

            train_set.to_csv(self.ingestionconfig.train_data_path , index=False , header=True)
            test_set.to_csv(self.ingestionconfig.test_data_path , index=False , header=True)
            logging.info("Data Ingestion Completed")

            return (
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()