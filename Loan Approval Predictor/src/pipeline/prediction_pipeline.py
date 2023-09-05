import os,sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import load_object

class PredictionPipeline:
    def __init__(self) :
        pass


    def predict(self,feature):
        preprocessor_path=os.path.join("artifacts/Data_Transformation","preprocessor.pkl")
        model_path=os.path.join("artifacts/model_trainer","model.pkl")

        process=load_object(preprocessor_path)
        model=load_object(model_path)

        scaled= process.transform(feature)
        pred=model.predict(scaled)

        return pred


class CustomClass:
    def __init__(self,
                 Gender:int,
                 Married:int,
                 Dependents:int,
                 Education:int,
                 Self_Employed:int,
                 ApplicantIncome:int,
                 CoapplicantIncome:int,
                 LoanAmount:int,
                 Loan_Amount_Term:int,
                 Credit_History:int,
                 Property_Area:int) :
        self.Gender=Gender,
        self.Married=Married,
        self.Dependents=Dependents,
        self.Education=Education,
        self.Self_Employed=Self_Employed,
        self.ApplicantIncome=ApplicantIncome,
        self.CoapplicantIncome=CoapplicantIncome,
        self.LoanAmount=LoanAmount,
        self.Loan_Amount_Term=Loan_Amount_Term,
        self.Credit_History=Credit_History,
        self.Property_Area=Property_Area



    def DataFrame(self):
            try:
                custom_input={
                    "Gender":[self.Gender],
                    "Married":[self.Married],
                    "Dependents":[self.Dependents],
                    "Education":[self.Education],
                    "Self_Employed":[self.Self_Employed],
                    "ApplicantIncome":[self.ApplicantIncome],
                    "CoapplicantIncome":[self.CoapplicantIncome],
                    "LoanAmount":[self.LoanAmount],
                    "Loan_Amount_Term":[self.Loan_Amount_Term],
                    "Credit_History":[self.Credit_History],
                    "Property_Area":[self.Property_Area]

                }

                data=pd.DataFrame(custom_input)

                return data
            except Exception as e:
                raise CustomException(e,sys)
            