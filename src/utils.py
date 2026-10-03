import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging
import dill

def save_object(filePath, preProcesser):
    '''this functiion will save the Processor in the provided filePath'''
    try:
        dir_name = os.path.dirname(filePath)
        os.makedirs(dir_name,exist_ok=True)

        with open(filePath,'wb') as fileObj:
            dill.dump(preProcesser,fileObj)

    except CustomException as e:
        raise (e,sys)