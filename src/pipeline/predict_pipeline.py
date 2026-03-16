import pandas as pd
import numpy as np
import sys
import os
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path="artifacts/model.pkl"
            preprocessor_path="artifacts/preprocessor.pkl"

            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):
        
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def dataframe_of_new_data(self):
        try:
            new_data={
            "gender":[self.gender],
            "race_ethnicity":[self.race_ethnicity],
            "parental_level_of_education":[self.parental_level_of_education],
            "lunch":[self.lunch],
            "test_preparation_course":[self.test_preparation_course],
            "reading_score":[self.reading_score],
            "writing_score":[self.writing_score]
            }

            new_df=pd.DataFrame(new_data)

            return new_df
        
        except Exception as e:
            raise CustomException(e,sys)