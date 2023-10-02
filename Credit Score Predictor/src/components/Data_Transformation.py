import os, sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder , StandardScaler
from src.utils import save_object
from imblearn.over_sampling import SMOTE


@dataclass
class DataTransformationConfig:
    PREPROCESSOR_PATH = os.path.join("artifacts/Data transformation","preprocessor.pkl")
    TRAIN_DF_PATH = os.path.join("artifacts/Data transformation/Data", "train.csv")
    TEST_DF_PATH = os.path.join("artifacts/Data transformation/Data", "test.csv")

class DataTransformation:
    def __init__(self) :
        self.DataTransformationConfig = DataTransformationConfig()

    def get_data_transform(self):
        try:

            Education = ['High School Diploma',"Associate's Degree","Bachelor's Degree","Master's Degree","Doctorate"]
            
            num_col = ['Age','Income','Number_of_Children']
            ordinal_col = ["Education"]
            cat_col = ['Gender','Marital_Status','Home_Ownership']

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=False))
                ],verbose=True
            )

            cat_pipeline = Pipeline(
                steps= [
                    ("ohe",OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))

                ],verbose= True
            )

            ord_pipeline = Pipeline(
                steps= [ 
                    ("ord",OrdinalEncoder(categories=[Education])),
                    ("scaler", StandardScaler(with_mean=False))

                ],verbose=True
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_col),
                    ("cat_pipeline", cat_pipeline, cat_col),
                    ("ord_pipeline", ord_pipeline, ordinal_col)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.info("Error Found in get data transform")
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self, train_data, test_data):
        try:
            logging.info("Train and Test data Has Read ")
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            train_df.rename(columns={"Number of Children":"Number_of_Children","Marital Status":"Marital_Status","Home Ownership":"Home_Ownership"},inplace=True)
            test_df.rename(columns={"Number of Children":"Number_of_Children","Marital Status":"Marital_Status","Home Ownership":"Home_Ownership"},inplace=True)

            
            

            target_col = "Credit Score"

            logging.info(f"columns ={train_df.columns}" )


            X_train = train_df.drop(target_col,axis= 1)
            y_train = train_df[target_col]

            X_test = test_df.drop(target_col, axis= 1)
            y_test = test_df[target_col]

            sm= SMOTE(random_state= 42)

            
            train_df.to_csv(self.DataTransformationConfig.TRAIN_DF_PATH, index= False)
            test_df.to_csv(self.DataTransformationConfig.TEST_DF_PATH, index= False)


            logging.info(f"shape of X {X_train.shape} and {X_test.shape}")
            logging.info(f"shape of  y {y_train.shape} and {y_test.shape}")

            preprocessor_obj = self.get_data_transform()

            #Transforming Data 
            X_train = preprocessor_obj.fit_transform(X_train)
            X_test = preprocessor_obj.transform(X_test)
            logging.info("transformation completed")
            X_res, y_res= sm.fit_resample(X_train, y_train)
            print('Class distribution before resampling:\n', y_train.value_counts())
            print("***************************************************************")
            print('Class distribution after resampling:\n', y_res.value_counts())



            #Concat
            train_arr = np.c_[X_res, np.array(y_res)]
            test_arr = np.c_[X_test, np.array(y_test)]

            logging.info("train_arr, test_arr Concatination completed")


            save_object(self.DataTransformationConfig.PREPROCESSOR_PATH,
                        obj= preprocessor_obj)
            
            logging.info("Saving Preprocessor object")

            return (
                train_arr,
                test_arr,
                self.DataTransformationConfig.PREPROCESSOR_PATH
            )


        except Exception as e:
            logging.info("Error Found in initiate_data_transformation")
            raise CustomException(e,sys)
    