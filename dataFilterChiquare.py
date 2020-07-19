#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:49:16 2019

@author: hoannguyen
"""

from numpy import mean
from numpy import std
from pandas import DataFrame
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(X_norm, y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')

def filterChiWithConfidenceLevel(data_all,confidence_level):
    
    if confidence_level == 90:
        sigma = 1.645
    elif confidence_level == 91:
        sigma = 1.695
    elif confidence_level == 92:
        sigma = 1.75
    elif confidence_level == 93:
        sigma = 1.81
    elif confidence_level == 94:
        sigma = 1.88
    elif confidence_level == 95:
        sigma = 1.96
    elif confidence_level == 96:
        sigma = 2.05
    elif confidence_level == 97:
        sigma = 2.17
    elif confidence_level == 98:
        sigma = 2.33
    elif confidence_level == 99:
        sigma = 2.58
    elif confidence_level == 99.73:
        sigma = 3
    elif confidence_level == 99.99366:
        sigma = 4
    elif confidence_level == 99.99932:
        sigma = 4.5
    
    dataexcept = []
    
    # calculate summary statistics
    data_mean, data_std = mean(data_all[:,:,0].reshape(-1)), std(data_all[:,:,0].reshape(-1))
    # identify outliers
    cut_off = data_std * sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off
    
    for i in range(data_all.shape[0]):
        for j in range(data_all.shape[1]):
            if data_all[i,j,0] < lower:
                data_all[i,j,0]= lower
                dataexcept.append(data_all[i,j,0])
            elif data_all[i,j,0] > upper:
                data_all[i,j,0] = upper
                dataexcept.append(data_all[i,j,0])
    
    df = DataFrame(dataexcept, columns= ['Power'])
    df.to_csv (r'export_datafilter.csv', index = None, header=True)
    return data_all
