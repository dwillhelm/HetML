#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   util.py
@Time    :   2022/04/27 00:28:33
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu
'''


# %%
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve

# get basepath
basedir = Path(os.path.dirname(os.path.abspath(__file__)))

class TrueVPredCV (): 
    def __init__(self): 
        self.train_size = 0.8
    
    def fit_all(self,pipe,y_true,y_pred,fig,ax):
        ax.scatter(y_true,y_pred,ec='k',alpha=0.8,)
        ax.grid(True,alpha=0.1)        
        upper = np.array([i.max() for i in [y_true,y_pred]]).max()
        lower = np.array([i.min() for i in [y_true,y_pred]]).min()
        upper = upper + 0.05*upper
        lower = lower - 0.05*upper
        ax.plot([lower,upper],[lower,upper],c='k',ls=':',zorder=-1)
        ax.set_xbound(lower,upper)
        ax.set_ybound(lower,upper)
        return fig,ax 
    
    def fit_plot(self,pipe,X,y,ax_labels=None,save_as=None): 
        xtrain,xtest, ytrain,ytest = train_test_split(X,y,train_size=0.8)
        pipe.fit(xtrain,ytrain)
        yhat_train = pipe.predict(xtrain)
        yhat_test  = pipe.predict(xtest)
        train_score = pipe.score(xtrain,ytrain).round(2)
        test_score = pipe.score(xtest,ytest).round(2)
        
        fig, ax = plt.subplots()
        ax.scatter(ytrain,yhat_train,ec='k',alpha=0.8,label=f'Training: {train_score}')
        ax.scatter(ytest,yhat_test,ec='k',alpha=0.8,label=f'Testing:{test_score}')
        ax.grid(True,alpha=0.1)
        
        if ax_labels is not None:         
            lx, ly = ax_labels
            ax.set_xlabel(lx)
            ax.set_ylabel(ly)           
        else: 
            ax.set_xlabel(r'$Y$')
            ax.set_ylabel(r'$\hat{Y}$')
        
        upper = np.array([i.max() for i in [ytrain,ytest,yhat_train,yhat_test]]).max()
        lower = np.array([i.min() for i in [ytrain,ytest,yhat_train,yhat_test]]).min()
        upper = upper + 0.05*upper
        lower = lower - 0.05*upper
        ax.plot([lower,upper],[lower,upper],c='k',ls=':',zorder=-1)
        ax.set_xbound(lower,upper)
        ax.set_ybound(lower,upper)
        ax.legend(loc=2) 
        
        if save_as is not None: 
            fig.tight_layout() 
            fig.savefig(save_as)
        else:
            return fig, ax 

class LearningCurve(): 
    def __init__(self,model,X,y,train_sizes=10,scoring=None,cv=None,n_jobs=-1):
        self.model = model
        self.X = X
        self.y = y 
        self.train_sizes = np.linspace(0.1,1.0,train_sizes)
        self.cv = None
        self.n_jobs = n_jobs 
        self.scoring = scoring 
    
    def fit_plot(self,fig,ax):
        ff, loc = 1,4 
        if self.scoring == 'mae': 
            self.scoring = 'neg_mean_absolute_error'
            ff = -1 
            loc=1
        train_sizes, train_scores, test_scores = \
        learning_curve(self.model, self.X, self.y, cv=self.cv, n_jobs=self.n_jobs,
                       train_sizes=self.train_sizes,
                       return_times=False,
                       scoring=self.scoring)   
        print(train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1) * ff 
        train_scores_std = np.std(train_scores, axis=1) * ff
        test_scores_mean = np.mean(test_scores, axis=1) * ff 
        test_scores_std = np.std(test_scores, axis=1) * ff 
        # Plot learning curve
        ax.grid()
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="tab:blue")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="tab:orange")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="tab:blue",
                     label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="tab:orange",
                     label="Cross-validation score")
        ax.legend(loc=loc)
        ax.grid(True,alpha=0.2)
        return fig,ax 