import os,sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion  
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass


if __name__=="__main__":
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()

    DataTransformation=DataTransformation()
    train_arr,test_arr,_= DataTransformation.initiate_data_transformation(train_data_path,test_data_path)

    model_training=ModelTrainer()
    model_training.initiate_model_trainer(train_arr,test_arr)

    