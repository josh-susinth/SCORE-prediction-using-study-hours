# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 22:31:21 2020

@author: Admin
"""

import pandas as pd
import numpy as np
import pickle


data=pd.read_csv(r"student_scores - student_scores.csv")

depen=data['Scores'].values.reshape(data['Scores'].shape[0],1)
indep=data['Hours']

from sklearn.linear_model import LinearRegression

reg=LinearRegression()
reg.fit(depen,indep)


pickle.dump(reg,open('model.pkl','wb'))


