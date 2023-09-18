import os, sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import load_object



class PredictionPipeline:
    def __init__(self) :
        pass

    def predict(self, features):
        try:
            preprocessor_path= os.path.join("Artifacts/Data_Transformation","preprocessor.pkl")
            model_path= os.path.join("Artifacts/Model_Trainer","model.pkl")

            processor= load_object(preprocessor_path)
            model= load_object(model_path)

            scaled=processor.transform(features)
            pred=model.predict(scaled)

            return pred
        
        except Exception as e:
            logging.info("Error Found in Prediction")
            raise CustomException(e,sys)
        

class CustomClass:
    def __init__(self,
                 cement: float,
                 blast_furnace_slag: float,
                 fly_ash: float,
                 water: float,
                 superplasticizer: float,
                 coarse_aggregate: float,
                 fine_aggregate: float,
                 age: float) :
        self.cement = cement
        self.blast_furnace_slag = blast_furnace_slag
        self.fly_ash = fly_ash
        self.water = water
        self.superplasticizer = superplasticizer
        self.coarse_aggregate = coarse_aggregate
        self.fine_aggregate = fine_aggregate
        self.age = age

    
    def get_Dataframe(self):
        try:
            custom_input = {
                "cement" : [self.cement],	
                "blast_furnace_slag": [self.blast_furnace_slag],
                "fly_ash" : [self.fly_ash],
                "water" : [self.water],
                "superplasticizer" : [self.superplasticizer],
                "coarse_aggregate" : [self.coarse_aggregate],
                "fine_aggregate" : [self.fine_aggregate],
                "age" : [self.age]

            }

            data = pd.DataFrame(custom_input)

            return data
        
        except Exception as e:
            logging.info("Error Found in Get Dataframe")
            raise CustomException(e,sys)
        