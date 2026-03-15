import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(obj,path):
    try:
        dir_name=os.path.dirname(path)

        os.makedirs(dir_name,exist_ok=True)

        with open(path,'wb')as f:
            pickle.dump(obj,f)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):

    try:
        report={}

        for name,model in models.items():

            print(f"{name} training...")

            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            report[name]=test_model_score
        return report

    except Exception as e:
        raise CustomException(e,sys)

    