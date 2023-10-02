import os, sys,pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score



def save_object(filepath , obj):
    try:
        dir_path = os.path.dirname(filepath)

        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,"wb") as file_path:
             pickle.dump(obj, file_path)
             
    except Exception as e:
            logging.info("Error Found in save_object ")
            raise CustomException(e,sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, classifier):
    try:
        report = {}

        for classifier_name, classifier in classifier.items():
            classifier.fit(X_train, y_train)

            # Make Predictions
            train_score = classifier.score(X_train, y_train)
            y_pred = classifier.predict(X_test)
            score = accuracy_score(y_pred, y_test)

            report[classifier_name] = score
            print(f"Classifier: {classifier_name}")
            print(f"Train score: {train_score:.2f}")
            print(f"Test score: {score:.2f}")

        best_model_name = max(report, key=report.get)
        print(f"The best model is: {best_model_name}")
        print(f"Accuracy of the best model: {report[best_model_name] :.2f}")

        return report

    except Exception as e:
            logging.info("Error Found in evaluate_model ")
            raise CustomException(e,sys)
     

def load_object(filepath):
     try:
          with open(filepath, 'rb') as object:
               return pickle.load(object)
          
     except Exception as e:
            logging.info("Error Found in load_object ")
            raise CustomException(e,sys)