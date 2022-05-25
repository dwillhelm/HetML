#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   predict.py
@Time    :   2022/05/25 09:40:23
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu
'''

import os
from pathlib import Path
import joblib
import pandas as pd 
import numpy as np 
from sklearn.metrics import mean_absolute_error
from hetml.data.dataloaders import load_featureset, quickload_datatables

import config

# get basepath
basedir = Path(os.path.dirname(os.path.abspath(__file__)))

def prediction_score(ytrue, ypred): 
    mae = mean_absolute_error(ytrue, ypred)
    print(f'\t-test MAE: {mae:.3f} {config.TARGET_UNIT}')

def load_pipe(): 
    path = Path(f'../trained_models/{config.MODEL_NAME}')
    pipe = joblib.load(path)
    return pipe

def predict(): 
    # load lookup and target datasets
    master, targets = quickload_datatables() 
    x,y,x_unlabeled = load_featureset(target=config.TARGET,stacktype=config.STACKTYPE) 

    # load trained pipeline
    pipeline = load_pipe() 
    
    # predict on labeld data (i.e. training data)
    print(f'\n\nPredicing {config.TARGET}:')
    y_pred = pipeline.predict(x)
    y_true = y.tolist() 
    predictions_1 = pd.DataFrame(
        np.column_stack((y_pred, y_true)),
        columns=['y_pred', 'y_true'], 
        index = y.index
    )  
    # get training score
    prediction_score(y_true, y_pred) 

    # predict on unlabeled data
    y_pred = pipeline.predict(x_unlabeled)
    y_true = [np.nan] * len(x_unlabeled)
    predictions_2 = pd.DataFrame(
        np.column_stack((y_pred, y_true)),
        columns=['y_pred', 'y_true'], 
        index = x_unlabeled.index
    )  

    print('\t-saving predictions to CSV file.\n')
    all_predictions = pd.concat((predictions_1, predictions_2))
    all_predictions.to_csv(f'{config.TARGET}_predictions.csv')

if __name__ == '__main__':
    
    predict() 