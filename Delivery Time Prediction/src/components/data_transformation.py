from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from src.utils import save_obj
from src.config.configuration import PREPROCESSING_OBJ_PATH,TRANSFORMED_TRAIN_FILE_PATH,TRANSFORMED_TEST_FILE_PATH,FEATURE_ENG_OBJ_PATH
# FE
# Data transformation


class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        """
        This class applies necessary Feature Engneering 
        """
        logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")


    
        
    
    def distance_numpy(self,df,lat1, lon1, lat2, lon2):
        p = np.pi/180
        a = 0.5 - np.cos((df[lat2]-df[lat1])*p)/2 + np.cos(df[lat1]*p) * np.cos(df[lat2]*p) * (1-np.cos((df[lon2]-df[lon1])*p))/2
        df['distance'] = 12742 * np.arcsin(np.sqrt(a))
        


    def transform_data(self,df):
        try:

            df.drop(['ID'],axis=1,inplace=True) 
            logging.info("dropping the Id column")


            logging.info("Creating feature on latitude nad longitude")
            self.distance_numpy(df,'Restaurant_latitude','Restaurant_longitude', 
                                'Delivery_location_latitude','Delivery_location_longitude')
            


            df.drop(['Delivery_person_ID','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude',
        'Order_Date','Time_Orderd','Time_Order_picked'],axis=1,inplace=True)
            
            


            logging.info(f'Train Dataframe Head: \n{df.head().to_string()}')

            return df
     



        except Exception as e:
            logging.info(" error in transforming data")
            raise CustomException(e, sys) from e 
        


    def fit(self,X,y=None):
        return self
    
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            transformed_df=self.transform_data(X)
                
            return transformed_df
        except Exception as e:
            raise CustomException(e,sys) from e



@dataclass
class DataTransformationConfig(): # first modofication
    preprocessor_obj_file_path=PREPROCESSING_OBJ_PATH
    transformed_train_path=TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_path=TRANSFORMED_TEST_FILE_PATH
    feature_eng_obj_path=FEATURE_ENG_OBJ_PATH


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')


            # defining the ordinal data ranking
            Road_traffic_density=['Low','Medium','High','Jam']
            Weather_conditions=['Sunny','Cloudy','Windy','Fog','Sandstorms','Stormy']
            

            # defining the categorical and numerical column
            categorical_column=['Type_of_order','Type_of_vehicle','Festival','City']
            ordinal_encod=['Road_traffic_density','Weather_conditions']
            numerical_column=['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition','multiple_deliveries','distance']
            

            # numerical pipeline

            numerical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='constant',fill_value=0)),
                ('scaler',StandardScaler(with_mean=False))
                ])



            # categorical pipeline

            categorical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
                ])




            # ordinal pipeline

            ordianl_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('ordinal',OrdinalEncoder(categories=[Road_traffic_density,Weather_conditions])),
                ('scaler',StandardScaler(with_mean=False))   
                ])
            
            # preprocessor

            preprocessor =ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_column),
                ('categorical_pipeline',categorical_pipeline,categorical_column),
                ('ordianl_pipeline',ordianl_pipeline,ordinal_encod)
                ])


            # returning preprocessor

            return preprocessor
        
            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("error in data transformation")
            raise CustomException(e,sys)
    ''' 
    def distance(self,lat1, lon1, lat2, lon2):
    # Convert latitude and longitude to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        R = 6371.0 # Earth's radius in km
        dist = R * c
    
        return dist
    '''

    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering())]) # 2nd modification
            return feature_engineering
        except Exception as e:
            raise CustomException(e,sys) from e


    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            #logging.info('Read train and test data completed')
            #logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            #logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')


            


# adding delivery city
            #train_df['Delivery_city']=train_df['Delivery_person_ID'].str.split('RES',expand=True)[0]
            #test_df['Delivery_city']=test_df['Delivery_person_ID'].str.split('RES',expand=True)[0]


            logging.info('Obtaining preprocessing object')



            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()

            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 20 + " Training data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Train Data ")

            train_df = fe_obj.fit_transform(train_df)
            logging.info(">>>" * 20 + " Test data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Test Data ")
            test_df = fe_obj.transform(test_df)

            train_df.to_csv("train_data.csv")
            test_df.to_csv("test_data.csv")
            logging.info(f"Saving csv to train_data and test_data.csv")

# preprocessing object

            preprocessing_obj = self.get_data_transformation_object()


            

# target column

            target_column_name = 'Time_taken (min)'
            #drop_columns = [target_column_name,'id']

            X_train = train_df.drop(columns=target_column_name,axis=1)
            y_train=train_df[target_column_name]

            X_test=test_df.drop(columns=target_column_name,axis=1)
            y_test=test_df[target_column_name]

            logging.info(f"shape of {X_train.shape} and {X_test.shape}")
            logging.info(f"shape of {y_train.shape} and {y_test.shape}")

            # Transforming using preprocessor obj
            
            X_train=preprocessing_obj.fit_transform(X_train)            
            X_test=preprocessing_obj.transform(X_test)
            logging.info("Applying preprocessing object on training and testing datasets.")
            logging.info(f"shape of {X_train.shape} and {X_test.shape}")
            logging.info(f"shape of {y_train.shape} and {y_test.shape}")
            

            logging.info("transformation completed")
            


            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            logging.info("train_arr, test_arr completed")

            

            logging.info("train arr , test arr")


            df_train= pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            logging.info("converting train_arr and test_arr to dataframe")
            #logging.info(f"Final Train Transformed Dataframe Head:\n{df_train.head().to_string()}")
            #logging.info(f"Final Test transformed Dataframe Head:\n{df_test.head().to_string()}")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path),exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transformed_train_path,index=False,header=True)

            logging.info("transformed_train_path")
            logging.info(f"transformed dataset columns : {df_train.columns}")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_path),exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transformed_test_path,index=False,header=True)


            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Preprocessor file saved")


            save_obj(
                file_path=self.data_transformation_config.feature_eng_obj_path,
                obj=fe_obj)
            logging.info("Feature eng file saved")

            
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)



        except Exception as e:
            logging.info("error in data transformation")
            raise CustomException(e,sys)
