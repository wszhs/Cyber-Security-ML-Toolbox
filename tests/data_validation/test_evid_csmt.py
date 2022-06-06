import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import matplotlib
import sage_zhs as sage
from csmt.data_validation.evidently import ColumnMapping
from csmt.data_validation.evidently.dashboard import Dashboard
from csmt.data_validation.evidently.dashboard.tabs import DataDriftTab
import matplotlib.pyplot as plt
import csmt.figure.visualml.visualml as vml
import pandas as pd
from csmt.get_model_data import get_datasets,parse_arguments,get_raw_datasets
import numpy as np

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm

    X,y,mask=get_raw_datasets(options)

    numerical_features=X.columns.values
    
    column_mapping = ColumnMapping(numerical_features=numerical_features)

    data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
    data_drift_dashboard.calculate(X.sample(1000, random_state=0), 
                                X.sample(1000, random_state=10), column_mapping=column_mapping)
    # data_drift_dashboard.show()

    data_drift_dashboard.save('tests/test_data_validation/ids17.html')