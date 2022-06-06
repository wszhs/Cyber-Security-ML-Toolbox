import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")

from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()

import pandas as pd
bc_df = pd.DataFrame(bc.data, columns=bc.feature_names)
bc_df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(bc_df, bc.target, test_size = 0.2, random_state = 31)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

from csmt.Interpretability.aix360.algorithms.rbm import FeatureBinarizer
fb = FeatureBinarizer(negations=True)
X_train_fb = fb.fit_transform(X_train)
X_test_fb = fb.transform(X_test)
X_train_fb['mean radius'][:8]

from csmt.Interpretability.aix360.algorithms.rbm import BRCGExplainer, BooleanRuleCG

boolean_model = BooleanRuleCG(silent=True)
explainer = BRCGExplainer(boolean_model)
explainer.fit(X_train_fb, Y_train)

Y_pred = explainer.predict(X_test_fb)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(f'Accuracy = {accuracy_score(Y_test, Y_pred)}')
print(f'Precision = {precision_score(Y_test, Y_pred)}')
print(f'Recall = {recall_score(Y_test, Y_pred)}')
print(f'F1 = {f1_score(Y_test, Y_pred)}')

e = explainer.explain()
isCNF = 'Predict Y=0 if ANY of the following rules are satisfied, otherwise Y=1:'
notCNF = 'Predict Y=1 if ANY of the following rules are satisfied, otherwise Y=0:'
print(isCNF if e['isCNF'] else notCNF)
print()
for rule in e['rules']:
    print(f'  - {rule}')