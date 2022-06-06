# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements the abstract estimators `TensorFlowEstimator` and `TensorFlowV2Estimator` for TensorFlow models.
"""
import logging
from typing import Any, Tuple, TYPE_CHECKING

import numpy as np

from csmt import config
from csmt.estimators.estimator import (
    BaseEstimator,
    LossGradientsMixin,
    NeuralNetworkMixin,
)


logger = logging.getLogger(__name__)


class TensorFlowEstimator(NeuralNetworkMixin, LossGradientsMixin, BaseEstimator):
    """
    Estimator class for TensorFlow models.
    """

    def __init__(self, **kwargs) -> None:
        """
        Estimator class for TensorFlow models.
        """
        self._sess: "tf.python.client.session.Session" = None
        super().__init__(**kwargs)

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs):
        """
        Perform prediction of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param batch_size: Batch size.
        :return: Predictions.
        :rtype: Format as expected by the `model`
        """
        return NeuralNetworkMixin.predict(self, x, batch_size=128, **kwargs)

    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the model of the estimator on the training data `x` and `y`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values.
        :type y: Format as expected by the `model`
        :param batch_size: Batch size.
        :param nb_epochs: Number of training epochs.
        """
        NeuralNetworkMixin.fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs)

    @property
    def sess(self) -> "tf.python.client.session.Session":
        """
        Get current TensorFlow session.

        :return: The current TensorFlow session.
        """
        if self._sess is not None:
            return self._sess
        else:
            raise NotImplementedError("A valid TensorFlow session is not available.")

    def loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :return: Loss values.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError


class TensorFlowV2Estimator(NeuralNetworkMixin, LossGradientsMixin, BaseEstimator):
    """
    Estimator class for TensorFlow v2 models.
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        TensorFlowV2Estimator._check_params(self)

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs):
        """
        Perform prediction of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param batch_size: Batch size.
        :return: Predictions.
        :rtype: Format as expected by the `model`
        """
        return NeuralNetworkMixin.predict(self, x, batch_size=128, **kwargs)

    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the model of the estimator on the training data `x` and `y`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values.
        :type y: Format as expected by the `model`
        :param batch_size: Batch size.
        :param nb_epochs: Number of training epochs.
        """
        NeuralNetworkMixin.fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs)

    def loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :return: Loss values.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        """
        Take a dictionary of parameters and apply checks before setting them as attributes.

        :param kwargs: A dictionary of attributes.
        """
        super().set_params(**kwargs)
        self._check_params()

    def _check_params(self) -> None:

        super()._check_params()