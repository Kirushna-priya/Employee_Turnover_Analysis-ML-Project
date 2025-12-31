
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''This function is responsible for data transformation'''
        try:
            # Define categorical columns
            department_column = ['department']
            salary_column = ['salary']

            # Pipeline for department (OneHotEncoder with drop_first)
            dept_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop='first', handle_unknown='ignore'))
                ]
            )

            # Pipeline for salary (OrdinalEncoder with explicit mapping)
            salary_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal_encoder", OrdinalEncoder(categories=[['low', 'medium', 'high']]))
                ]
            )

            # Combine pipelines in ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("dept_pipeline", dept_pipeline, department_column),
                    ("salary_pipeline", salary_pipeline, salary_column)
                ]
            )

            logging.info("Categorical columns encoding completed")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train & test data completed')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "left"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying SMOTE to training data")

            # Match notebook parameters
            smote = SMOTE(sampling_strategy=0.8, random_state=42)
            input_feature_train_arr, target_feature_train_df = smote.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )

            logging.info(f"Resampled training data shape: {input_feature_train_arr.shape}, {target_feature_train_df.shape}")

            # Convert sparse matrix to dense array
            input_feature_train_arr = input_feature_train_arr.toarray()
            input_feature_test_arr = input_feature_test_arr.toarray()

            # Convert Series to 2D NumPy array
            target_feature_train_df = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_df = target_feature_test_df.values.reshape(-1, 1)

            train_arr = np.concatenate((input_feature_train_arr, target_feature_train_df), axis=1)
            test_arr = np.concatenate((input_feature_test_arr, target_feature_test_df), axis=1)

            print("Final train_arr shape:", train_arr.shape)
            print("Final test_arr shape:", test_arr.shape)

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)