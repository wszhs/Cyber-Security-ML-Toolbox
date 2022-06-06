# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the classifier `XGBoostClassifier` for XGBoost models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from copy import deepcopy
import json
import logging
import os
import pickle
from typing import List, Optional, Union, Tuple, TYPE_CHECKING

import numpy as np

from csmt.estimators.classification.classifier import ClassifierDecisionTree
from csmt.utils import to_categorical
from csmt import config

logger = logging.getLogger(__name__)


class XGBoostClassifier(ClassifierDecisionTree):
    """
    Wrapper class for importing XGBoost models.
    """

    def __init__(
        self,
        model: Union["xgboost.Booster", "xgboost.XGBClassifier", None] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        nb_features: Optional[int] = None,
        nb_classes: Optional[int] = None,
    ) -> None:
        """
        Create a `Classifier` instance from a XGBoost model.

        :param model: XGBoost model.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param nb_features: The number of features in the training data. Only used if it cannot be extracted from
                             model.
        :param nb_classes: The number of classes in the training data. Only used if it cannot be extracted from model.
        """
        from xgboost import Booster, XGBClassifier

        if not isinstance(model, Booster) and not isinstance(model, XGBClassifier):
            raise TypeError("Model must be of type xgboost.Booster or xgboost.XGBClassifier.")

        super().__init__(
            model=model,
            clip_values=clip_values
        )
        self._input_shape = (nb_features,)
        self._nb_classes = nb_classes

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    def fit(self, x: np.ndarray, y: np.ndarray,X_val,y_val,**kwargs) -> None:

        self.model.fit(x, y, **kwargs)
        

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """

        # Apply preprocessing
        x_preprocessed=x
        predictions = self._model.predict_proba(x_preprocessed)

        return predictions

    def _get_nb_classes(self, nb_classes: Optional[int]) -> int:
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        """
        from xgboost import Booster, XGBClassifier

        if isinstance(self._model, Booster):
            try:
                return int(len(self._model.get_dump(dump_format="json")) / self._model.n_estimators)
            except AttributeError:
                if nb_classes is not None:
                    return nb_classes
                raise NotImplementedError(
                    "Number of classes cannot be determined automatically. "
                    + "Please manually set argument nb_classes in XGBoostClassifier."
                ) from AttributeError

        if isinstance(self._model, XGBClassifier):
            return self._model.n_classes_

        return -1

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        """
        if path is None:
            full_path = os.path.join(config.ART_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(full_path + ".pickle", "wb") as file_pickle:
            pickle.dump(self._model, file=file_pickle)

    def get_trees(self) -> List["Tree"]:
        trees = list()
        return trees

