import os, sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from src.utils import evaluate_model, save_object 
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join("Artifacts/Model_Trainer", "model.pkl")

class ModelTrainer:
    def __init__(self) :
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:,-1]
                )
            
            models = {
                # "Random Forest": RandomForestRegressor(),
                # "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                
                "AdaBoost Regressor": AdaBoostRegressor()}


            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }

            }

            logging.info("Model Evaluation is Initiated")
            model_report:dict= evaluate_model(X_train,y_train,X_test,y_test,models,params)

            # To Get best model Score
            best_model_score= max(sorted(model_report.values()))

            # To get Name of Best Model
            best_model_name= list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]


            logging.info(f"Best Model Found, Name of the Model is {best_model},\n, Score is {best_model_score}")

            print("*****************************************************************************************")    
            print(f"Best Model Found, Name of the Model is {best_model},\n,  Score is {best_model_score}")


            save_object(file_path=self.model_trainer_config.trained_model_path,
                        obj= best_model)
            logging.info("Model file is Being Saved")


        except Exception as e:
            logging.info("Error Found in Model Trainer")
            raise CustomException (e,sys)