import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")

from sklearn.datasets import load_boston
boston = load_boston()

import pandas as pd
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(boston_df, boston.target, test_size = 0.25, random_state = 31)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

from csmt.Interpretability.aix360.algorithms.rbm import FeatureBinarizer
fb = FeatureBinarizer(negations=True)
X_train_fb = fb.fit_transform(X_train)
X_test_fb = fb.transform(X_test)
# print(X_train_fb['CRIM'][:10])
from csmt.Interpretability.aix360.algorithms.rbm import GLRMExplainer, LinearRuleRegression
linear_model = LinearRuleRegression()
explainer = GLRMExplainer(linear_model)
explainer.fit(X_train_fb, Y_train)

Y_pred = explainer.predict(X_test_fb)

from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, max_error
print(f'R2 Score = {r2_score(Y_test, Y_pred)}')
print(f'Explained Variance = {explained_variance_score(Y_test, Y_pred)}')
print(f'Mean abs. error = {mean_absolute_error(Y_test, Y_pred)}')
print(f'Max error = {max_error(Y_test, Y_pred)}')

result=explainer.explain()
print(result)