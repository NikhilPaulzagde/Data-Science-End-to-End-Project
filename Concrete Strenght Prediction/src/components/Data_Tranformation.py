import os, sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose  import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.utils import save_object
from src.components.Data_Ingestion import DataIngestion


@dataclass
class DataTransformationConfig:
    processor_obj_path = os.path.join("Artifacts/Data_Transformation","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            numerical_columns=["cement","blast_furnace_slag","fly_ash","water","superplasticizer","coarse_aggregate","fine_aggregate" ,"age"]

            num_pipeline = Pipeline(
                steps = [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ],verbose=True
            )
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor= ColumnTransformer(
                [("num_pipeline",num_pipeline,numerical_columns)]
            )

            return preprocessor
            
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        

    def remove_outlier( self,col , df):
        try:
            Q1=df[col].quantile(0.25)
            Q3=df[col].quantile(0.75)

            IQR=Q3-Q1
            
            upper_limit = Q3 + (IQR*1.5)
            lower_limit = Q1 - (IQR*1.5)
            
            df.loc[(df[col]>upper_limit),col] = upper_limit
            df.loc[(df[col]<lower_limit),col] = lower_limit
            
            return df

        except Exception as e:
            raise CustomException(e,sys)



    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df= pd.read_csv(train_data_path)
            test_df= pd.read_csv(test_data_path)

            logging.info("Read train and test data completed")

            train_df=train_df.drop_duplicates()
            test_df=test_df.drop_duplicates()

            train_df['age']=train_df["age"].astype("float64")

            numerical_columns=["cement","blast_furnace_slag","fly_ash","water","superplasticizer","coarse_aggregate","fine_aggregate","age"]

            for col in numerical_columns:
                self.remove_outlier(col=col, df=train_df)

            logging.info("Outlier capped on our train data")

            for col in numerical_columns:
                self.remove_outlier(col=col, df=test_df)

            logging.info("Outlier capped on our test data")


            preprocessor_obj=self.get_data_transformation()

            target_col="concrete_compressive_strength"

            logging.info("Spliting Train data into Dependent and Independent feature")
            input_feauture_train= train_df.drop(target_col,axis=1)
            target_feauture_train= train_df[target_col]
            print("Input train feature",input_feauture_train.dtypes,target_feauture_train.shape)
          

            logging.info("Spliting Test data into Dependent and Independent feature")
            input_feauture_test= test_df.drop(target_col,axis=1)
            target_feauture_test= test_df[target_col]
            print("Test Feature",input_feauture_test.shape,target_feauture_test.shape)

            # Apply Transformation on our Train data
            input_train_array= preprocessor_obj.fit_transform(input_feauture_train)
            input_test_array= preprocessor_obj.transform(input_feauture_test)
            logging.info("Data has Transformed")
            print("Input Train array ",input_train_array.dtype,
                  "\n target feature train",target_feauture_train.dtype)

            #Concat Dependent and Independent feature
            train_array= np.c_[input_train_array,np.array(target_feauture_train)]
            test_array= np.c_[input_test_array,np.array(target_feauture_test)]
            logging.info("Data has been Concat")
            print(train_array.shape,test_array.shape)

            save_object(file_path= self.data_transformation_config.processor_obj_path,
                        obj= preprocessor_obj)
            logging.info("Preprocessor file is Saved Sucessfully")
            
            return ( train_array, 
                     test_array, 
                     self.data_transformation_config.processor_obj_path)

        except Exception as e:
            raise CustomException(e,sys)
        


if __name__ == "__main__":
    Data_Ingestion = DataIngestion()
    train_data_path,test_data_path = Data_Ingestion.initiate_data_ingestion()

    Data_transformation= DataTransformation()
    train_array,test_array,_= Data_transformation.initiate_data_transformation(train_data_path,test_data_path)