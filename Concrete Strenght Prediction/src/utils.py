import os, sys
from src.exception import CustomException
from src.logger import logging
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info("Error in Saving Object")
        raise CustomException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        train_report={}
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]


            GS=GridSearchCV(model,para,cv=5)
            GS.fit(X_train,y_train)

            model.set_params(**GS.best_params_)
            model.fit(X_train,y_train)

            #make prediction
            y_pred=model.predict(X_test)
            test_model_accuracy=r2_score(y_test,y_pred)

            train_y_pred=model.predict(X_train)
            train_model_accuracy=r2_score(y_train,train_y_pred)

            

            report[list(models.values())[i]]=test_model_accuracy
            train_report[list(models.values())[i]]=train_model_accuracy


            print("Train Model R2 score,:",train_report)

            return report
    except Exception as e:
        logging.info("Error Found in Evaluation")
        raise CustomException(e,sys)
    


def load_object(filepath):
    try:
        with open(filepath, "rb") as object:
            return pickle.load(file=object)
    except Exception as e:
        logging.info("Error Found in Loading Object")
        raise CustomException(e,sys)
    
