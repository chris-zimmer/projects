# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 07:23:23 2020

@author: zimme
"""


import numpy as np
import pandas as pd

#Importing the training data
train_df = pd.read_csv('C:/Users/zimme/Documents/Kaggle Competitions/train.csv')
train_df.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1, inplace = True)
X = train_df.iloc[:,1:].values
y = train_df.iloc[:,0].values


#Importing the testing data
predictions_df = pd.read_csv('C:/Users/zimme/Documents/Kaggle Competitions/test.csv')

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

#Pipelines
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numerical_transformer = SimpleImputer(strategy = 'mean')
categorical_transformer = Pipeline(steps = [('impute',SimpleImputer(strategy = 'most_frequent')),
                                            ('onehot',OneHotEncoder(handle_unknown = 'ignore'))])
preprocessor = ColumnTransformer(transformers = [('num', numerical_transformer, [2]),
                                                 ('cat', categorical_transformer, [1,6])])

#Random Forest ensemble model, training on the split data:
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 100, random_state = 0)

from sklearn.metrics import mean_absolute_error
the_pipeline = Pipeline(steps=[('preprocess',preprocessor),('model',model)])
the_pipeline.fit(X_train,y_train)
preds = the_pipeline.predict(X_test)
score = mean_absolute_error(y_test, preds)

#XGBoost:
from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=1)
the_pipeline = Pipeline(steps=[('preprocess',preprocessor),('my_model',my_model)])
the_pipeline.fit(X_train,y_train)
preds = the_pipeline.predict(X_test)
score_XG = mean_absolute_error(y_test, preds)

#XGBoost ensemble model, trained on the entire dataset. Fit to the actual test data:
predictions_df.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1, inplace = True)
x_pred = predictions_df.iloc[:,:].values
pred = the_pipeline.predict(x_pred)


pred = (pred > 0.5)
pred = pd.DataFrame(pred)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(pred.values) 
y = pd.DataFrame(y)
y.to_csv("Titanic_XGBoost.csv", index = False)










