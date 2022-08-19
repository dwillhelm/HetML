#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   dataloaders.py
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu
'''

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from hetml.data.preprocessing import FilterSteps

# get basepath
basedir = Path(os.path.dirname(os.path.abspath(__file__)))
datadir = basedir.joinpath('../../data')

def quickload_datatables(): 
    # rel pth to data
    # master df 
    df1 = pd.read_pickle(datadir.joinpath('tables/bilayer_masterlist.pkl'))
    df1 = df1.set_index('hhuid').rename_axis('HUID')
    # target df
    df2 = pd.read_pickle(datadir.joinpath('targets/processed/bilayer_target_properties.pkl'))
    df2 = df2.rename_axis('HUID')
    # # featurespace
    # df3 = pd.read_pickle(datadir.joinpath('features/raw/blfeatures-set3.pkl'))
    print('returning --> master | targets')
    return df1, df2  
    
class LoadFeatureSets():
    def __init__(self): 
        fsets = ['bilayer_banddos_features.pkl',
                 'bilayer_c2db_basefeatures.pkl',
                 'bilayer_c2db_basefeatures_poly.pkl',
                 'bilayer_c2db_p2_basefeatures.pkl',
                 'bilayer_c2db_p3_basefeatures.pkl',
                 'bilayer_composition_features.pkl',
                 'bilayer_composition_features-full_magpie.pkl',
                 'bilayer_composition_features-full_matminer.pkl',
                 'bilayer_xbm_siteproperties.pkl']

        fsets = dict(zip(range(len(fsets)),fsets))
        fsets[100] = 'blfeatures-set2.pkl'
        fsets[101] = 'bilayer_c2db_p0_basefeatures.pkl'

        self.fsets = fsets 
        
    def fit_from_index(self, dfs_idx): 
        dfs = [self.fsets[i] for i in dfs_idx]
        dfs = [basedir.joinpath('../../data/features/raw').joinpath(i) for i in dfs]
        dfs = [pd.read_pickle(i) for i in dfs]
        df = self._merge_dfs(dfs)
        return df
        
    def _merge_dfs(self,dfs): 
        from functools import reduce
        df_merged = reduce(
            lambda  left,right: pd.merge(left,right,
                                         left_index=True,
                                         right_index=True,
                                         ), dfs) 
        return df_merged

def load_featureset(target='Egap', stacktype='AUB'): 
    """
    returns processed/final features, targets, unlabeled data
    """
    # get reference datasets
    master, targets = quickload_datatables() 
    # load default raw features
    fs0 = LoadFeatureSets().fit_from_index([2,0,8,7])
    # apply filtering steps
    x,y,x0 = FilterSteps(master, targets, fs0).fit(target, stacktype)
    
    return x,y,x0


