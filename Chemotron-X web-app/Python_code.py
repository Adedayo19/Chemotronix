# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 22:17:38 2021

@author: Professor
"""

import pandas as pd
import pickle
data = pd.read_csv('CE_train.csv')
data= data.drop(['Unnamed: 0'], axis=1)
data['country'].replace(['Brazil','China','EU27 & UK','France','Germany','India','Italy','Japan','ROW','Russia','Spain','UK','US','WORLD'],[0.18,4.85,1.42,0.13,0.30,1.09,0.14,0.49,4.41,0.68,0.11,0.16,.21,15.36],inplace= True)
data.drop("timestamp",axis=1, inplace=True)
data['sector'].replace(['Power','Industry','Ground Transport','Residential','International Aviation','Domestic Aviation'],[6,5,4,3,2,1],inplace= True)
y = data.value
x = data.drop('value', axis =1)
from catboost import CatBoostRegressor #Categorical boosting
model= CatBoostRegressor(depth=10, learning_rate=0.1, iterations=100)
model.fit(x,y)
pickle.dump(model,open("model.pkl","wb"))
model=pickle.load(open("model.pkl","rb"))
