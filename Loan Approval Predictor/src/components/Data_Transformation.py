import pandas as pd
import numpy as np
import os,sys
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from imblearn.over_sampling import RandomOverSampler


@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path=os.path.join("artifacts/Data_Transformation","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation=DataTransformationConfig()


    def get_data_transform_obj(self):
        try:
            logging.info("Data Transformation has Started")

            numerical_feature=["Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area"] 

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_feature)
            ])

            logging.info("Imputation and Scaling is completed")
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)    
     
    def remove_outlier(self,col,df):
        try:
            Q1=df[col].quantile(0.25)
            Q3=df[col].quantile(0.75)
            IQR=Q3-Q1

            upper_limit=Q3+IQR*1.5
            lower_limit=Q1-IQR*1.5

            df.loc[(df[col]>upper_limit),col]=upper_limit
            df.loc[(df[col]<lower_limit),col]=lower_limit

            return df
        
        except Exception as e:
            logging.info("Outliers Handled")
            raise CustomException(e,sys)
         

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            col = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

            for i in col:
              
                self.remove_outlier(col=i, df=train_data)

            logging.info("Outlier capped on our train data")

            for i in col:
               
                self.remove_outlier(col=i, df=test_data)

            logging.info("Outlier capped on our test data")

            preprocessor_obj = self.get_data_transform_obj()

            target_column = "Loan_Status"
            drop_column = [target_column]

            logging.info("Splitting train data into dependent and independent feature")
            input_feature_train_data = train_data.drop(drop_column, axis=1)
            target_feature_train_data = train_data[target_column]
            print("target_feature_train_data:",target_feature_train_data.value_counts())

            logging.info("Splitting test data into dependent and independent feature")
            input_feature_test_data = test_data.drop(drop_column, axis=1)
            target_feature_test_data = test_data[target_column]
            

            # Apply Random Oversampling to balance the class distribution
            ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
            input_feature_train_data_resampled, target_feature_train_data_resampled = ros.fit_resample(
                input_feature_train_data, target_feature_train_data)
            
            print("target_feature_test_data_resampled:",target_feature_train_data_resampled.value_counts())

            # Apply transformations on the resampled training data
            input_train_arr = preprocessor_obj.fit_transform(input_feature_train_data_resampled)

            # Apply transformations on the test data
            input_test_arr = preprocessor_obj.transform(input_feature_test_data)

            # Concatenate input_train_arr and target_feature of train and test data
            train_array = np.c_[input_train_arr, np.array(target_feature_train_data_resampled)]
            test_array = np.c_[input_test_arr, np.array(target_feature_test_data)]

            save_object(file_path=self.data_transformation.preprocess_obj_file_path, obj=preprocessor_obj)

            return (train_array, test_array, self.data_transformation.preprocess_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)
