import os, sys
from src.components.Data_ingestion import DataIngestion
from src.components.Data_Transformation import DataTransformation
from src.components.Model_Trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

if __name__=="__main__":
        obj=DataIngestion()
        train_data_path,test_data_path=obj.initiate_data_ingestion()

        Data_transform=DataTransformation()
        train_array, test_array,_=Data_transform.initiate_data_transformation(train_data_path,test_data_path)

        model_trainer=ModelTrainer()
        model_trainer.initiate_model_trainer(train_array, test_array)
