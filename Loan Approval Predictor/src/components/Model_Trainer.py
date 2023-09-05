import pandas as pd
import numpy as np
import os,sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object , evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join("artifacts/model_trainer","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train,y_train,X_test,y_test= (
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:,-1]
            )

            logging.info("Data Has Been Splitted")

            model={
                "Random Forest":RandomForestClassifier(),
                #"Decision Tree":DecisionTreeClassifier()
            }

            params={
                    "Random Forest":{
                        'n_estimators': [100, 200, 300],      
                        'max_depth': [None,1,2,3,4,5,6,7,8,9,10 ,20 ,30] ,    
                        'min_samples_split': [2, 5,10],     
                        'min_samples_leaf': [1, 2, 4]     },

                    # "Decision Tree":{
                    #             'criterion': ['gini', 'entropy','log_loss'],
                    #             'max_depth': [1,2,3,4,5,10],
                    #             'min_samples_split': [2,5,7],
                    #             'min_samples_leaf': [1,2,3,4,5,6,7,8,9],
                    #             'max_features': [None, 'sqrt', 'log2'],  # Maximum number of features to consider for splitting
                    #             'min_impurity_decrease': [0.0, 0.1, 0.2],  # Minimum impurity decrease required for a split
              
            }      
              
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=model,params=params)

            logging.info("Model Has Been Evaluated")

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=model[best_model_name]

            logging.info(f"Best Model Found, Model name is {best_model_name} Accuracy Score is {best_model_score}")
            print(f'{best_model_name} & Score --> ', best_model_score)

            save_object(file_path=self.model_trainer_config.train_model_file_path,
                       obj=best_model )


        except Exception as e:
            raise CustomException(e,sys)

