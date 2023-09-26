import os,sys
from src.exception import CustomException
from src.logger import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass
from src.utils import evaluate_model,save_object


@dataclass
class ModelTrainerConfig:
    MODEL_TRAINER_PATH = os.path.join("Artifacts/Model Trainer", "model.pkl")


class ModelTrainer:
    def __init__(self) :
        self.ModelTrainerConfig = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            X_train, X_test, y_train, y_test=(
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            ) 

            logging.info("Models training has initiated")
            #Defining models 

            classifiers = {
                    'K-NearestNeighbors': KNeighborsClassifier(),
                    'SupportVectorMachine': SVC(kernel='linear', C=1.0),
                    'DecisionTree': DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1),
                    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1)
                }
            
            logging.info("Models Evaluation has initiated")

            model_report:dict = evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, classifier=classifiers)

            #To get best_model score
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(classifiers.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = classifiers[best_model_name]  # Assign the actual model object, not just the name

            print(f"Best model found, Model Name is {best_model_name},accuracy score : {best_model_score}")
            print("*"*90)

            logging.info(f"Best model found, Model Name is {best_model_name},accuracy score : {best_model_score}")

            save_object(filepath=self.ModelTrainerConfig.MODEL_TRAINER_PATH,
                        obj=best_model)

        except  Exception as e:
            logging.info("Error Found in initate model trainer")
            raise CustomException(e,sys)
