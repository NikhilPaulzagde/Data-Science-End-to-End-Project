import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import  SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils import save_object 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join("artifacts/model_trainer","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spitting Data into Dependent and Independent feature")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info("Models training has initiated")
            # Define your models and parameters for classification
            model = {
                     "Decision Tree": DecisionTreeClassifier(),
                     "Random Forest": RandomForestClassifier(),
                     "Logistic Regression": LogisticRegression(),
                }

            params = {
                    "Decision Tree": {
                        "criterion": ["gini", "entropy"],
                        "splitter": ["best", "random"],
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                    },
                    "Random Forest": {
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                        "criterion": ["gini", "entropy"],
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                    },
                    
                    "Logistic Regression": {
                        "penalty": ["l1", "l2"],
                        "C": [0.01, 0.1, 1, 10],
                        "solver": ["liblinear", "saga"],
                    },
                  
                   
                 
                }



          
            logging.info("Model evaluation has initiated")
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=model,params=params)

            #To get best model fro report Dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name= list(model.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model= model[best_model_name]

            print(f"Best model found, Model Name is {best_model_name},accuracy score : {best_model_score}")
            print("*"*90)

            logging.info(f"Best model found, Model Name is {best_model_name},accuracy score : {best_model_score}")

            save_object(file_path=self.model_trainer_config.train_model_file_path,
                        obj=best_model)


        except Exception as e:
            raise CustomException(e, sys)