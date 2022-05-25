#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   pipeline.py
@Time    :   2022/05/25 09:53:24
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu
'''

import os
from pathlib import Path
import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor

import config

# get basepath
basedir = Path(os.path.dirname(os.path.abspath(__file__)))

## Models/Algorithms

# LASSO feature selection model
lasso_fs = SelectFromModel(estimator=
        LassoCV(
            n_alphas=1000,
            max_iter=4500,
            n_jobs=config.N_JOBS,
            cv=5, 
            normalize=False) #data is already normalized/scaled
) 

# stacked regressor
basemodels = [
    ('GBR',    GradientBoostingRegressor()) ,
    ('Ridge',  RidgeCV() ) ,
    ('SVR',    SVR(kernel='linear')) ,
    ('KRR',    KernelRidge(kernel='poly'))
    ]
model = StackingRegressor(
            estimators=basemodels, 
            final_estimator=LinearRegression())


## Pipelines and transformers

num_pipe = Pipeline(steps=[
    ('scale_features', StandardScaler() )
])

preprocessing_pipe = ColumnTransformer(transformers=[
    ('num_features', num_pipe, config.NUM_FEATS )
], remainder='drop')


lasso_featsel_pipe = Pipeline(steps=[
    ('preprocessing',   preprocessing_pipe),
    ('lasso_feat_sel',  lasso_fs),
])

full_pipeline = Pipeline(steps=[
    ('preprocessing',       preprocessing_pipe),
    ('lasso_feature_sel',   lasso_fs),
    ('model',               model)
])


def get_lasso_features(pipe, x_data): 
    # get lasso coefs from pipeline
    estimator = pipe['lasso_feature_sel'].estimator_
    coefs = estimator.coef_ 
    # make into dataframe
    coefs = pd.DataFrame(coefs, columns=['lasso_coefs'], index=x_data.columns)
    coefs = coefs[coefs.abs() > 0].dropna() # drop zero magnitude coefs
    coefs = coefs.sort_values('lasso_coefs')
    return coefs

def save_lasso_features(pipe, x_data, filename='lasso_coefs.csv'): 
    # get lasso features
    coefs = get_lasso_features(pipe, x_data)
    # save to file
    coefs.to_csv(filename)