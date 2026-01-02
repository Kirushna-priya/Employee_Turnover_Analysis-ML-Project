import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
        satisfaction_level:float,
        number_project:int,
        average_montly_hours: int,
        time_spend_company: int,
        department: str,
        salary: str):

        self.satisfaction_level = satisfaction_level

        self.number_project = number_project

        self.average_monthly_hours = average_montly_hours

        self.time_spend_company = time_spend_company

        self.department = department

        self.salary = salary

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "satisfaction_level": [self.satisfaction_level],
                "number_project": [self.number_project],
                "average_montly_hours": [self.average_monthly_hours],
                "time_spend_company": [ self.time_spend_company],
                "department": [ self.department],
                "salary": [self.salary]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
    