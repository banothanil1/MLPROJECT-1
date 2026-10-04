import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from src.utils import evaluate_model
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class ModelTrainingConfig:
    ModelTrainingConfigPath = os.path.join('Artifact','model.pkl')

class ModelTraining:
    def __init__(self):
        self.trained_model_path = ModelTrainingConfig()

    def model_train(self, train_arr, test_arr):
        '''this model is used to train the model using the train data and test data'''
        try:
            models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Ridge": Ridge(),
                    "Lasso": Lasso(),
                    "Elastic Net": ElasticNet(),
                    "SVR": SVR(),
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "AdaBoost": AdaBoostRegressor(),
                    "KNN": KNeighborsRegressor(),
                    "XGBoost": XGBRegressor(),
                    "CatBoost": CatBoostRegressor()
                }
            X_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]
            X_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]

            model_report = evaluate_model(X_train,y_train,X_test,y_test,models)

            best_score = max(list(model_report.values()))

            if best_score <= 0.6:
                raise CustomException("No Best Model Found")
                    
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]
            
            logging.info(f"Best Model Name is {best_model_name} and best score is {best_score}")
            
            save_object(filePath= self.trained_model_path.ModelTrainingConfigPath, preProcesser= best_model_name)
            
            logging.info(f'Model converted to pickle file and saved to artifact and path is{self.trained_model_path.ModelTrainingConfigPath}')

            return [best_model_name,best_score]
            
        except Exception as e:
            raise CustomException(e,sys)