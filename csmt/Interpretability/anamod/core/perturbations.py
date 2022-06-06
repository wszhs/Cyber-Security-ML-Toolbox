"""Classes for managing perturbations"""

from abc import ABC
from itertools import permutations
from math import factorial

import numpy as np

from anamod.core import constants


# pylint: disable = invalid-name
class PerturbationFunction(ABC):
    """Perturbation function base class"""
    def __init__(self, *args):
        pass

    def operate(self, X):
        """Operate on input"""


class Zeroing(PerturbationFunction):
    """Replace input values by zeros (deprecated)"""
    def operate(self, X):
        X[:] = 0
        return X


class Permutation(PerturbationFunction):
    """Permute first axis of data array"""
    def __init__(self, rng, num_elements, num_permutations, *args):
        super().__init__(*args)
        # If the number of instances is less than the number of permutations, we need to enumerate all permutations.
        # Else, we just shuffle
        self.pool = None
        if num_permutations >= num_elements:
            total_permutations = factorial(num_elements)
            # TODO: Probability of collisions is ~sqrt(num_elements) with permutations, so ideally we may want to
            # enumerate permutations even if the number of possible permutations is greater than the sample count
            if num_permutations >= total_permutations:
                self.pool = permutations(range(num_elements))
                self.pool.__next__()  # First permutation is the original order
        self._rng = rng

    def operate(self, X):
        if self.pool is not None:
            idx = self.pool.__next__()  # Caller needs to catch StopIteration
            return X[idx, ...]

        self._rng.shuffle(X)
        return X


class PerturbationMechanism(ABC):
    """Performs perturbation"""
    def __init__(self, perturbation_fn, perturbation_type,
                 rng, num_elements, num_permutations):
        # pylint: disable = too-many-arguments
        assert issubclass(perturbation_fn, PerturbationFunction)
        self._perturbation_fn = perturbation_fn(rng, num_elements, num_permutations)
        assert perturbation_type in {constants.ACROSS_INSTANCES, constants.WITHIN_INSTANCE}
        self._perturbation_type = perturbation_type

    def perturb(self, X, feature, *args, **kwargs):
        """Perturb feature for input data and given feature(s)"""
        size = feature.size
        if size == 0:
            return X  # No feature(s) to be perturbed
        if size == 1:
            idx = feature.idx[0]  # To enable fast view-based indexing for singleton features
        else:
            idx = feature.idx
        X_hat = np.copy(X)
        return self._perturb(X_hat, idx, *args, **kwargs)

    def _perturb(self, X_hat, idx, *args, **kwargs):
        """Perturb feature for input data and given feature indices"""


class PerturbMatrix(PerturbationMechanism):
    """Perturb input arranged as matrix of instances X features"""
    def _perturb(self, X_hat, idx, *args, **kwargs):
        perturbed_slice = self._perturbation_fn.operate(X_hat[:, idx])
        X_hat[:, idx] = perturbed_slice
        return X_hat


class PerturbTensor(PerturbationMechanism):
    """Perturb input arranged as tensor of instances X features X time"""
    def _perturb(self, X_hat, idx, *args, **kwargs):
        timesteps = kwargs.get("timesteps", slice(None))
        axis0 = slice(None)  # all sequences
        axis1 = idx  # features to be perturbed
        axis2 = timesteps  # timesteps to be perturbed
        if self._perturbation_type == constants.WITHIN_INSTANCE:
            X_hat = np.transpose(X_hat)
            axis0, axis2 = axis2, axis0  # swap sequence and timestep axis for within-instance permutation
        perturbed_slice = self._perturbation_fn.operate(X_hat[axis0, axis1, axis2])
        if self._perturbation_type == constants.WITHIN_INSTANCE and timesteps == slice(None) and np.isscalar(idx):
            # Basic indexing - view was perturbed, so no assignment needed
            X_hat = X_hat.base
            assert perturbed_slice.base is X_hat
            return X_hat
        X_hat[axis0, axis1, axis2] = perturbed_slice
        return np.transpose(X_hat) if self._perturbation_type == constants.WITHIN_INSTANCE else X_hat


PERTURBATION_FUNCTIONS = {constants.ACROSS_INSTANCES: {constants.ZEROING: Zeroing, constants.PERMUTATION: Permutation},
                          constants.WITHIN_INSTANCE: {constants.ZEROING: Zeroing, constants.PERMUTATION: Permutation}}
PERTURBATION_MECHANISMS = {constants.HIERARCHICAL: PerturbMatrix, constants.TEMPORAL: PerturbTensor}
