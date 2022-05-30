#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   preprocessing.py
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

from hetml.data.util import final_feature_columns

# get basepath
basedir = Path(os.path.dirname(os.path.abspath(__file__)))
datadir = basedir.joinpath('../../data')

def get_monolayer_workfunctions(): 
    tdf = pd.read_pickle(datadir.joinpath('dft/processed/dft_workfunctions.pkl'))
    tdf = pd.read_csv(datadir.joinpath('dft/processed/dft_workfunctions0.csv'),index_col=0)
    return tdf

class FilterSteps(): 
    """Apply some filtering and feature engineering steps to a dataset"""
    def __init__(self,master, targets, fs): 
        self.master = master
        self.targets = targets
        self.fs = fs
        
    def fit(self, target, stacktype): 
        ## setup 
        print('\nPreprocessing Steps:')
        fs_ = self.fs.copy() 
        targets_ = self.targets.copy() 
        master = self.master
        print(f'Feature Space Dim: {fs_.shape}')
        print(f'Targets Dim: {targets_.shape}')
        
        ## add anderson rule class 
        print("\t-building Anderson's Rule classes")
        colmap = {'I':int(1), 'II':int(2), 'III':int(3), 'other':int(0)}
        fs_['band_alignment'] = master.reindex(fs_.index).band_alignment.map(colmap)
        fs_.band_alignment = fs_.band_alignment.astype(int)

        ## add stacktype class 
        print("\t-building stacking configuration classes")
        colmap = {'I':int(1), 'II':int(2), 'III':int(3), 'other':int(0)}
        fs_['stacktype'] = master.reindex(fs_.index).stacktype

        # filter out by stacktype
        if stacktype != 'AUB': 
            fs_ = fs_[fs_.stacktype==stacktype]
            fs_ = fs_.drop(columns=['stacktype'])
            targets_ = targets_[targets_.stacktype==stacktype]
            print(f'\t-using {stacktype} bilayers, dropping features: {fs_.shape}')
            print(f'\t-using {stacktype} bilayers, dropping targets: {targets_.shape}')
        elif stacktype == 'AUB': 
            fs_.stacktype = fs_.stacktype.map({'AB':int(0), 'AA':int(1)})
            fs_.stacktype = fs_.stacktype.astype(int)
            print(f'\t-using {stacktype} bilayers, transforming to binary classes')

        # add monolayer workfunction features. 
        tdf = get_monolayer_workfunctions()
        avg_wf, min_wf, max_wf, delta_wf = [],[],[],[]
        for huid in fs_.index: 
            uid0, uid1 = master.loc[huid][['uid_bot','uid_top']]
            wf0 = tdf.loc[uid0].work_function
            wf1 = tdf.loc[uid1].work_function
            wfs = np.array([wf0,wf1])
            avg_wf.append(0.5 * (wf0 + wf1))
            min_wf.append(wfs.min())
            max_wf.append(wfs.max())
            delta_wf.append(abs(wf0 - wf1))

        fs_['avg_wf'] = avg_wf 
        fs_['min_wf'] = min_wf 
        fs_['max_wf'] = max_wf
        fs_['delta_wf'] = delta_wf

        # manual filter
        fs_ = fs_.reindex(columns=final_feature_columns)
        print(f'\t-dropping feature columns: {fs_.shape}')

        # filter targets by conduction type
        if target in ['Egap','Egap_dir','IE','EA']: 
            targets_ = targets_[targets_.is_metal == False]
            print(f'\t-dropping metal bilayers: {targets_.shape}')
        else: 
            pass 
        
        # filter Type III according to target
        if target in ['Egap','Egap_dir','IE','EA']:
            targets_ = targets_[targets_.band_alignment.isin(['I','II'])]
            print(f'\t-dropping Type III bilayers: {targets_.shape}')
            print(targets_.band_alignment.value_counts())
        
        # filter targets by threshold
        targets_ = targets_[targets_.ILD >= 2.5]
        print(f'\t-dropping bilayers w/ ILD < 2.5: {targets_.shape}')
        targets_ = targets_[targets_.Ebind < 40.0 ]
        print(f'\t-dropping bilayers w/ Eb > 40 meV: {targets_.shape}')
        targets_ = targets_[targets_.charge_transfer < 1.0]
        print(f'\t-dropping bilayers w/ charge transf. > 1 |e|: {targets_.shape}')
        
        # get dummy variables for anderson gap class (i.e. one-hot-encoding)
        fs_ = pd.get_dummies(fs_,columns=['band_alignment'])
        fs_.band_alignment_1 = fs_.band_alignment_1.astype(int) 
        fs_.band_alignment_2 = fs_.band_alignment_2.astype(int) 
        fs_.band_alignment_3 = fs_.band_alignment_3.astype(int) 
        if target in ['Egap','Egap_dir','IE','EA']:
            fs_ =fs_.drop(columns=['band_alignment_3'])
        print(f'\t-one-hot-encod band alignment: {fs_.shape}')

        print(f'\nFinal --> Feature Space Dim: {fs_.shape}')
        print(f'Final --> Targets Dim: {targets_.shape}')

        fs_.stacktype = fs_.stacktype.astype(int)

        ## consolidate training data and unlabled data
        y  = targets_[target]
        x  = fs_.reindex(y.index)
        x0 = fs_.drop(x.index)

        print(f'\n{target = }  {stacktype = }')
        print(f'x dim: {x.shape}')
        print(f'y dim: {y.shape}')      
        print(f'unlabeled-X dim: {x0.shape}\n')  
        print(f'\nAll Features (Pre-Feature Selection) (p={len(x.columns)}):')
        print(x.columns.tolist())

        return x,y,x0 