
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox/")
import datetime
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import json

from sklearn import datasets, ensemble, model_selection
from scipy.stats import anderson_ksamp

from csmt.data_validation.evidently.dashboard import Dashboard
from csmt.data_validation.evidently.pipeline.column_mapping import ColumnMapping
from csmt.data_validation.evidently.dashboard.tabs import DataDriftTab, NumTargetDriftTab, RegressionPerformanceTab
from csmt.data_validation.evidently.options import DataDriftOptions
from csmt.data_validation.evidently.model_profile import Profile
from csmt.data_validation.evidently.model_profile.sections import DataDriftProfileSection, RegressionPerformanceProfileSection

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

content = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip").content
with zipfile.ZipFile(io.BytesIO(content)) as arc:
    raw_data = pd.read_csv(arc.open("hour.csv"), header=0, sep=',', parse_dates=['dteday']) 

raw_data.index = raw_data.apply(lambda row: datetime.datetime.combine(row.dteday.date(), datetime.time(row.hr)),
                                axis=1)

target = 'cnt'
prediction = 'prediction'
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
categorical_features = ['season', 'holiday', 'workingday', ]#'weathersit']

reference = raw_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
current = raw_data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00']

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    reference[numerical_features + categorical_features],
    reference[target],
    test_size=0.3
)

regressor = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)

regressor.fit(X_train, y_train)

preds_train = regressor.predict(X_train)
preds_test = regressor.predict(X_test)

X_train['target'] = y_train
X_train['prediction'] = preds_train

X_test['target'] = y_test
X_test['prediction'] = preds_test

column_mapping = ColumnMapping()

column_mapping.target = 'target'
column_mapping.prediction = 'prediction'
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features

regression_perfomance_dashboard = Dashboard(tabs=[RegressionPerformanceTab()])
regression_perfomance_dashboard.calculate(X_train.sort_index(), X_test.sort_index(), 
                                          column_mapping=column_mapping)

regression_perfomance_dashboard.save('tests/test_data_validation/bicycle.html')

