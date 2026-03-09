import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utilis import load_object

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/pipeline/
ROOT_DIR = os.path.join(BASE_DIR, '..')           # project root

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join(ROOT_DIR, 'artifacts', 'model.pkl')
            preprocessor_path = os.path.join(ROOT_DIR, 'artifacts', 'proprocessor.pkl')  # ✅ your filename
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
