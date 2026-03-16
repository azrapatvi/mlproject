import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(obj,path):
    try:
        dir_name=os.path.dirname(path)

        os.makedirs(dir_name,exist_ok=True)

        with open(path,'wb')as f:
            pickle.dump(obj,f)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params, n_iter=20):
    try:
        report = {}
        best_models = {}  # to store tuned models

        for name, model in models.items():
            print(f"Training and tuning {name}...")

            rs=RandomizedSearchCV(estimator=model,param_distributions=params[name],n_iter=30,cv=4,verbose=2,n_jobs=-1)

            rs.fit(X_train, y_train)

            best_model = rs.best_estimator_

            y_test_pred = best_model.predict(X_test)

            score = r2_score(y_test, y_test_pred)

            report[name] = score

            best_models[name] = best_model

        return report, best_models

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(path):
    try:
        with open(path,'rb')as f:
            return(pickle.load(f))
        
    except Exception as e:
        raise CustomException(e,sys)