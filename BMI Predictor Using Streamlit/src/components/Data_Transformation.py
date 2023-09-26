import pandas as pd
import numpy as np
import os,sys
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils import save_object



@dataclass
class DataTransformerConfig:
    PROCESSOR_OBJ_FILE_PATH = os.path.join("Artifacts/Data Transformation", "Preprocessor.pkl") 


class DataTransformation:
    def __init__(self) :
        self.DataTransformerConfig = DataTransformerConfig()


    def get_data_transformation(self):
        try:
            cat_col = ['Gender']
            num_col = ['Height','Weight']

            cat_pipeline = Pipeline(
                steps= [ 
                    ("ohe", OneHotEncoder())
                ]
            )

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Categorical columns: {cat_col}")
            logging.info(f"Numerical columns: {num_col}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_col),
                    ("cat_pipeline", cat_pipeline, cat_col)
                ],verbose= True
            )

            return preprocessor

        except Exception as e:
            logging.info("Error Found in get data transformation")
            raise CustomException(e,sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = train_df.drop_duplicates()
            test_df = test_df.drop_duplicates()

            logging.info(f"Train data : {train_df.duplicated().sum()} Test data : {test_df.duplicated().sum()}")
            print(f"Train data : {train_df.duplicated().sum()}  Test data : {test_df.duplicated().sum()}")
            logging.info("Duplicated Value removed")

            target_col = "Index"

            X_train = train_df.drop(target_col, axis= 1 )
            y_train = train_df[target_col]

            X_test = test_df.drop(target_col, axis=1)
            y_test = test_df[target_col]

            logging.info("Applying Processor Object on X_train and X_test")

            processor_obj = self.get_data_transformation()

            X_train_arr = processor_obj.fit_transform(X_train)
            X_test_arr = processor_obj.transform(X_test)

            #Concatinating the Dataset

            X_train_array = np.c_[X_train_arr, np.array(y_train)]
            X_test_array = np.c_[X_test_arr, np.array(y_test)]

            logging.info(" Saving preprocessor object")

            save_object(filepath= self.DataTransformerConfig.PROCESSOR_OBJ_FILE_PATH,
                        obj= processor_obj)
            
            return (
                X_train_array,
                X_test_array,
                self.DataTransformerConfig.PROCESSOR_OBJ_FILE_PATH
                
            )


        except Exception as e:
            logging.info("Error Found in initiate data transformation")
            raise CustomException(e,sys)
       