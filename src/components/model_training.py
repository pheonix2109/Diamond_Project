import numpy as np
import pandas as pd

# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
from src.utils import print_evaluated_results
from src.utils import model_metrics

from dataclasses import dataclass
import sys
import os
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models)
            

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                logging.info('Best model has r2 Score less than 60%')
                raise CustomException('No Best Model Found')
            
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')


            logging.info('Hyperparameter tuning started for XGBoost')
           
            # hyperparameter tuning for XGBOOST MODEL

            xgb = XGBRegressor()

            # Parameters
            params = {
            'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
            'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
            'min_child_weight' : [ 1, 3, 5, 7 ],
            'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
            'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ],
            'n_estimators':[300,400,500,600]
            }

            rs_xgb=RandomizedSearchCV(xgb,param_distributions=params,scoring='r2',n_jobs=-1,cv=5)
            rs_xgb.fit(X_train, y_train)
           
            # Print the tuned parameters and score
          
            print(f'Best XGBoost parameters : {rs_xgb.best_params_}')
            print(f'Best XGBoost Score : {rs_xgb.best_score_}')
            
            print('\n====================================================================================\n')

            best_xgb = rs_xgb.best_estimator_

            logging.info('Hyperparameter tuning complete for XGBoost.')


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_xgb
            )
            logging.info('Model pickle file saved')
            # Evaluating Ensemble Regressor (Voting Classifier on test data)
            y_test_pred = best_xgb.predict(X_test)

            mae, rmse, r2 = model_metrics(y_test, y_test_pred)
            logging.info(f'Test MAE : {mae}')
            logging.info(f'Test RMSE : {rmse}')
            logging.info(f'Test R2 Score : {r2}')
            logging.info('Final Model Training Completed\n')
            
            return mae, rmse, r2 



        except Exception as e:
            raise CustomException(e,sys)