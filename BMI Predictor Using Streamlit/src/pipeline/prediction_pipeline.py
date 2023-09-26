import os,sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import pandas as pd

class PredictionPipeline:
    def __init__(self) :
        pass


    def get_predict(self, features):
        try:
            PROCESSOR_PATH = os.path.join("Artifacts/Data Transformation","Preprocessor.pkl")
            MODEL_PATH = os.path.join("Artifacts/Model Trainer","model.pkl")

            preprocessor = load_object(PROCESSOR_PATH)
            model= load_object(MODEL_PATH)

            scaled = preprocessor.transform(features)
            pred = model.predict(scaled)

            return pred
        
        except Exception as e:
            logging.info("Error Found in Predict")
            raise CustomException(e,sys)


class CustomClass:
    def __init__(self,
                 Gender:int,
                 Height:int,
                 Weight:int) :
        
        self.Gender = Gender
        self.Height = Height
        self.Weight = Weight


    def get_dataframe(self):
        try:
            custom_input = {
                "Gender" : [self.Gender],
                "Height" : [self.Height],
                "Weight" : [self.Weight]
            }

            data = pd.DataFrame(custom_input)

            return data
        
        except Exception as e:
            logging.info("Error Found in Get dataframe")
            raise CustomException(e,sys)

