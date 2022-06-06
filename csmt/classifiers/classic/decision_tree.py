'''
Author: your name
Date: 2021-03-24 16:38:32
LastEditTime: 2021-07-10 19:03:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/classic/decision_tree.py
'''
from csmt.classifiers.abstract_model import AbstractModel
from csmt.estimators.classification.scikitlearn import SklearnClassifier
from sklearn import tree

class DecisionTree(AbstractModel):

    def __init__(self,
            input_size,
            output_size,
            split_criterion='entropy',
            splitter='best',
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=0
        ):
        model = tree.DecisionTreeClassifier(
            criterion=split_criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        self.classifier = SklearnClassifier(model=model,clip_values=(0,1))