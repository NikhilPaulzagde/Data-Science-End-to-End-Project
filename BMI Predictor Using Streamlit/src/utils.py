import os,sys
from src.exception import CustomException
from src.logger import logging
import pickle
from sklearn.metrics import accuracy_score


def save_object(filepath,obj):
    try:
        dir_name = os.path.dirname(filepath)

        os.makedirs(dir_name, exist_ok=True)

        with open(filepath, "wb") as file_path_obj:
            pickle.dump(obj,file_path_obj) 
    except Exception as e:
        logging.info("Error Found in Saving Object")
        raise CustomException(e,sys)


def evaluate_model(X_train, X_test, y_train, y_test, classifier):
    try:
        report = {}

        for classifier_name, classifier in classifier.items():
            classifier.fit(X_train, y_train)

            # Make predictions
            train_score = classifier.score(X_train, y_train)
            y_pred = classifier.predict(X_test)
            score = accuracy_score(y_pred, y_test)

            report[classifier_name] = score
            print(f"Classifier: {classifier_name}")
            print(f"Train score: {train_score:.2f}")
            print(f"Test score: {score:.2f}")

                # Find the best model
            best_model_name = max(report, key=report.get)
            print(f"The best model is: {best_model_name}")
            print(f"Accuracy of the best model: {report[best_model_name]:.2f}")

        return report


    except Exception as e:
        logging.info("Error Found in Evaluate Model")
        raise CustomException(e,sys)

def load_object(filepath):
    try:
        with open(filepath,'rb') as objt:
            return pickle.load(objt) 
        
    except Exception as e:
        logging.info("Error Found in Load Object")
        raise CustomException(e,sys)

def load_object(filepath):
    try:
        with open(filepath,"rb") as file_objt:
            return pickle.load(file_objt)
    except Exception as e:
        raise CustomException(e,sys)
    