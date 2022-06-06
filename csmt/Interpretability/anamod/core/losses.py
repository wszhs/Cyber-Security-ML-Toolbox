"""Loss functions"""

from abc import ABC
from inspect import currentframe, getframeinfo
import sys

import numpy as np

from anamod.core import constants

TARGET_VALUES = {constants.LABELS, constants.BASELINE_PREDICTIONS}


class LossFunction(ABC):
    """Loss function base class"""
    @staticmethod
    def loss(y_true, y_pred):
        """Return vector of losses given true and predicted model values over a list of instances"""


class QuadraticLoss(LossFunction):
    """Quadratic loss function"""
    @staticmethod
    def loss(y_true, y_pred):
        return (y_true - y_pred)**2


class AbsoluteDifferenceLoss(LossFunction):
    """Absolute difference loss function (like quadratic loss, but scales differently)"""
    @staticmethod
    def loss(y_true, y_pred):
        return np.abs(y_true - y_pred)


class ZeroOneLoss(LossFunction):
    """0-1 loss"""
    @staticmethod
    def loss(y_true, y_pred):
        y_true = (y_true > 0.5)
        y_pred = (y_pred > 0.5)
        return (y_true != y_pred).astype(np.int32)


class BinaryCrossEntropy(LossFunction):
    """Binary cross-entropy"""
    fp_warned = False

    @staticmethod
    def fp_warn(err, flag):
        """Warn once if zero encountered in np.log"""
        # pylint: disable = unused-argument
        if not BinaryCrossEntropy.fp_warned:
            frameinfo = getframeinfo(currentframe())
            sys.stderr.write(f"Warning: {frameinfo.filename}: {frameinfo.lineno}: 0 encountered in np.log; "
                             "ensure that model predictions are probabilities\n")
            BinaryCrossEntropy.fp_warned = True

    np.seterrcall(fp_warn.__func__)

    @staticmethod
    def loss(y_true, y_pred):
        assert all(y_pred >= 0) and all(y_pred <= 1)
        with np.errstate(invalid="call", divide="call"):
            losses = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        losses[np.isnan(losses)] = 0  # to handle indeterminate case where y_pred[i] = y_true[i]
        losses[np.isinf(losses)] = 23  # to handle case where y_pred[i] = 1 - y_true[i]; -log(1e-10) ~ 23
        return losses


LOSS_FUNCTIONS = {constants.QUADRATIC_LOSS: QuadraticLoss,
                  constants.ABSOLUTE_DIFFERENCE_LOSS: AbsoluteDifferenceLoss,
                  constants.BINARY_CROSS_ENTROPY: BinaryCrossEntropy,
                  constants.ZERO_ONE_LOSS: ZeroOneLoss}


class Loss():
    """Compute losses given true and predicted model values over a list of instances"""
    def __init__(self, loss_function, targets):
        self._loss_fn = LOSS_FUNCTIONS[loss_function].loss
        self._targets = targets

    def loss_fn(self, predictions):
        """Return loss vector"""
        return self._loss_fn(self._targets, predictions)
