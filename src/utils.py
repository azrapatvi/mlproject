import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.exception import CustomException

def save_object(obj,path):
    try:
        dir_name=os.path.dirname(path)

        os.makedirs(dir_name,exist_ok=True)

        with open(path,'wb')as f:
            pickle.dump(obj,f)

    except Exception as e:
        raise CustomException(e,sys)

    