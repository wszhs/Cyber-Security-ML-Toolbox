
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox/")

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml

from csmt.data_validation.evidently import ColumnMapping
from csmt.data_validation.evidently.dashboard import Dashboard
from csmt.data_validation.evidently.dashboard.tabs import DataDriftTab

data = fetch_openml(name='adult', version=2, as_frame='auto')
df = data.frame
df.head()

df['num_feature_with_3_values'] = np.random.choice(3, df.shape[0])
df['num_feature_with_2_values'] = np.random.choice(2, df.shape[0])

numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'num_feature_with_3_values', 'num_feature_with_2_values']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'class']
column_mapping = ColumnMapping(numerical_features=numerical_features, categorical_features=categorical_features)

data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
data_drift_dashboard.calculate(df.sample(1000, random_state=0), 
                               df.sample(1000, random_state=10), column_mapping=column_mapping)
# data_drift_dashboard.show()

data_drift_dashboard.save('tests/test_data_validation/adult.html')