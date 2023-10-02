import os, sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import pandas as pd

class PredictionPipeline:
    def __init__(self) :
        pass

    def predict (self, feature):
        try:
            processor_path = os.path.join("artifacts/Data transformation","preprocessor.pkl")
            model_path = os.path.join("artifacts/Model Trainer","model.pkl")

            preprocessor = load_object(processor_path)
            model = load_object(model_path)

            scaled = preprocessor.transform(feature)
            pred = model.predict(scaled)

            return pred

        except Exception as e:
            logging.info("Error Found in predict")
            raise CustomException(e,sys)


class Customclass:
    def __init__(self,Age: int,
                 Gender: str,
                 Income: int,
                 Education: str,
                 Marital_Status: str,
                 Number_of_Children: int,
                 Home_Ownership: str ) :
        self.Age = Age
        self.Gender = Gender
        self.Income = Income
        self.Education = Education
        self.Marital_Status = Marital_Status
        self.Number_of_Children = Number_of_Children
        self.Home_Ownership = Home_Ownership


    def get_dataframe(self):
        try:
            custom_data = {
                "Age" : [self.Age ],
                "Gender" : [self.Gender],
                "Income" : [self.Income],
                "Education" : [self.Education],
                "Marital_Status" : [self.Education],
                "Number_of_Children" : [self.Number_of_Children],
                "Home_Ownership" : [self.Home_Ownership]


            }

            data = pd.DataFrame(custom_data)

            return data


        except Exception as e:
            logging.info("Error Found in get_dataframe")
            raise CustomException(e,sys)

