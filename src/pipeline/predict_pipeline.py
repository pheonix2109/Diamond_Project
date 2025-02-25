import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model_path = 'artifacts\model.pkl'
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 carat:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)