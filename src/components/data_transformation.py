import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #applying pipline to columns :[ (name, transformer, columns) ]
from sklearn.impute import SimpleImputer #missing values handling :fill with median here
from sklearn.pipeline import Pipeline #stey-by-step execution :steps[('name_of_step', transformer_object)]
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utilis import save_object
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifacts','preprocessor.pkl') 
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            numerical_columns=['math score','reading score','writing score']
            categoical_columns=['gender','race/ethnicity','parental level of education','lunch','test preparation course']
            num_pipeline=Pipeline( 
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ("Scaler",StandardScaler())
                ]
            )
            logging.info('Numerical columns imputation completed')
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler())
                ]
            )
            
            logging.info('Categorical columns imputation completed')
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categoical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Read train and test data completed')
            logging.info("obtaning preprocessor object")
            preproceesing_obj=self.get_data_transformer_object()
            target_column_name='math score'
            numerical_columns=['reading score','writing score']
        

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr=preproceesing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preproceesing_obj.transform(input_feature_test_df) 
            logging.info("Preprocessing completed")
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)] 
            logging.info("Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preproceesing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)



