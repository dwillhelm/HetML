#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   train_pipe.py
@Time    :   2022/05/25 09:40:37
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu
'''

import os
from pathlib import Path
import joblib

from hetml.data.dataloaders import load_featureset, quickload_datatables

import config
from pipeline import full_pipeline, save_lasso_features

# get basepath
basedir = Path(os.path.dirname(os.path.abspath(__file__)))

def save_pipe(pipeline, filename):
    save_path = Path('../trained_models').joinpath(filename)
    joblib.dump(pipeline, save_path)

def train_pipeline(): 
    """Train ML Model"""
    # load lookup and target datasets
    master, targets = quickload_datatables() 
    x,y,_ = load_featureset(target=config.TARGET,stacktype=config.STACKTYPE) 

    # fit final model and save
    print('\n\nRunning Training Pipeline:')
    full_pipeline.fit(x,y)
    lasso_features = save_lasso_features(full_pipeline, x)
    print('\t-training succesful, saving model\n')
    save_pipe(full_pipeline, config.MODEL_NAME)

if __name__ == '__main__':

    train_pipeline() 