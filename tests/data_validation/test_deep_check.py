
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox/")
import numpy as np
# General imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from csmt.data_validation.deepchecks.tabular.datasets.classification import iris

# Load Data
iris_df = iris.load_data(data_format='Dataframe', as_train_test=False)
label_col = 'target'
df_train, df_test = train_test_split(iris_df, stratify=iris_df[label_col], random_state=0)

# Train Model
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(df_train.drop(label_col, axis=1), df_train[label_col])

from csmt.data_validation.deepchecks.tabular import Dataset

# We explicitly state that this dataset has no categorical features, otherwise they will be automatically inferred
# If the dataset has categorical features, the best practice is to pass a list with their names

ds_train = Dataset(df_train, label=label_col, cat_features=[])
ds_test =  Dataset(df_test,  label=label_col, cat_features=[])

from csmt.data_validation.deepchecks.tabular.suites import full_suite

suite = full_suite()

suite_result=suite.run(train_dataset=ds_train, test_dataset=ds_test, model=rf_clf)

suite_result.save_as_html('tests/data_validation/check.html')

