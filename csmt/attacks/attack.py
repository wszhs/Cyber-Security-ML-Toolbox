
"""
This module implements the abstract base classes for all attacks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import logging
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from csmt.exceptions import EstimatorError


logger = logging.getLogger(__name__)


class input_filter(abc.ABCMeta):
    """
    Metaclass to ensure that inputs are ndarray for all of the subclass generate and extract calls
    """

    def __init__(cls, name, bases, clsdict):
        """
        This function overrides any existing generate or extract methods with a new method that
        ensures the input is an `np.ndarray`. There is an assumption that the input object has implemented
        __array__ with np.array calls.
        """

        def make_replacement(fdict, func_name):
            """
            This function overrides creates replacement functions dynamically
            """

            def replacement_function(self, *args, **kwargs):
                if len(args) > 0:
                    lst = list(args)

                if "x" in kwargs:
                    if not isinstance(kwargs["x"], np.ndarray):
                        kwargs["x"] = np.array(kwargs["x"])
                else:
                    if not isinstance(args[0], np.ndarray):
                        lst[0] = np.array(args[0])

                if "y" in kwargs:
                    if kwargs["y"] is not None and not isinstance(kwargs["y"], np.ndarray):
                        kwargs["y"] = np.array(kwargs["y"])
                elif len(args) == 2:
                    if not isinstance(args[1], np.ndarray):
                        lst[1] = np.array(args[1])

                if len(args) > 0:
                    args = tuple(lst)
                return fdict[func_name](self, *args, **kwargs)

            replacement_function.__doc__ = fdict[func_name].__doc__
            replacement_function.__name__ = "new_" + func_name
            return replacement_function

        replacement_list = ["generate", "extract"]
        for item in replacement_list:
            if item in clsdict:
                new_function = make_replacement(clsdict, item)
                setattr(cls, item, new_function)


class Attack(abc.ABC, metaclass=input_filter):
    """
    Abstract base class for all attack abstract base classes.
    """

    attack_params: List[str] = list()
    _estimator_requirements: Optional[Union[Tuple[Any, ...], Tuple[()]]] = None

    def __init__(self, estimator):
        """
        :param estimator: An estimator.
        """
        if self.estimator_requirements is None:
            raise ValueError("Estimator requirements have not been defined in `_estimator_requirements`.")

        if not all(t in type(estimator).__mro__ for t in self.estimator_requirements):
            raise EstimatorError(self.__class__, self.estimator_requirements, estimator)

        self._estimator = estimator

    @property
    def estimator(self):
        return self._estimator

    @property
    def estimator_requirements(self):
        return self._estimator_requirements

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: A dictionary of attack-specific parameters.
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        self._check_params()

    def _check_params(self) -> None:
        pass


class EvasionAttack(Attack):
    """
    Abstract base class for evasion attack classes.
    """

    def __init__(self, **kwargs) -> None:
        self._targeted = False
        super().__init__(**kwargs)

    @abc.abstractmethod
    def generate(  # lgtm [py/inheritance/incorrect-overridden-signature]
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        evasion attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted
               or not. This parameter is only used by some of the attacks.
        :return: An array holding the adversarial examples.
        """
        raise NotImplementedError

    @property
    def targeted(self) -> bool:
        """
        Return Boolean if attack is targeted. Return None if not applicable.
        """
        return self._targeted

    @targeted.setter
    def targeted(self, targeted) -> None:
        self._targeted = targeted


class PoisoningAttack(Attack):
    """
    Abstract base class for poisoning attack classes
    """

    def __init__(self, classifier: Optional["CLASSIFIER_TYPE"]) -> None:
        """
        :param classifier: A trained classifier (or none if no classifier is needed)
        """
        super().__init__(classifier)

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y=Optional[np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate poisoning examples and return them as an array. This method should be overridden by all concrete
        poisoning attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y:  Target labels for `x`. Untargeted attacks set this value to None.
        :return: An tuple holding the (poisoning examples, poisoning labels).
        """
        raise NotImplementedError


class PoisoningAttackTransformer(PoisoningAttack):
    """
    Abstract base class for poisoning attack classes that return a transformed classifier.
    These attacks have an additional method, `poison_estimator`, that returns the poisoned classifier.
    """

    def __init__(self, classifier: Optional["CLASSIFIER_TYPE"], **kwargs) -> None:
        """
        :param classifier: A trained classifier (or none if no classifier is needed)
        """
        super().__init__(classifier)

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y=Optional[np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate poisoning examples and return them as an array. This method should be overridden by all concrete
        poisoning attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y:  Target labels for `x`. Untargeted attacks set this value to None.
        :return: An tuple holding the (poisoning examples, poisoning labels).
        :rtype: `(np.ndarray, np.ndarray)`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def poison_estimator(self, x: np.ndarray, y: np.ndarray, **kwargs) -> "CLASSIFIER_TYPE":
        """
        Returns a poisoned version of the classifier used to initialize the attack
        :param x: Training data
        :param y: Training labels
        :return: A poisoned classifier
        """
        raise NotImplementedError


class PoisoningAttackBlackBox(PoisoningAttack):
    """
    Abstract base class for poisoning attack classes that have no access to the model (classifier object).
    """

    def __init__(self):
        """
        Initializes black-box data poisoning attack.
        """
        super().__init__(None)  # type: ignore

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate poisoning examples and return them as an array. This method should be overridden by all concrete
        poisoning attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y:  Target labels for `x`. Untargeted attacks set this value to None.
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        raise NotImplementedError


class PoisoningAttackWhiteBox(PoisoningAttack):
    """
    Abstract base class for poisoning attack classes that have white-box access to the model (classifier object).
    """

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate poisoning examples and return them as an array. This method should be overridden by all concrete
        poisoning attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted
               or not. This parameter is only used by some of the attacks.
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        raise NotImplementedError


class ExtractionAttack(Attack):
    """
    Abstract base class for extraction attack classes.
    """

    @abc.abstractmethod
    def extract(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> "CLASSIFIER_TYPE":
        """
        Extract models and return them as an ART classifier. This method should be overridden by all concrete extraction
        attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted
               or not. This parameter is only used by some of the attacks.
        :return: ART classifier of the extracted model.
        """
        raise NotImplementedError


class InferenceAttack(Attack):
    """
    Abstract base class for inference attack classes.
    """

    def __init__(self, estimator):
        """
        :param estimator: A trained estimator targeted for inference attack.
        :type estimator: :class:`.art.estimators.estimator.BaseEstimator`
        """
        super().__init__(estimator)

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer sensitive properties (attributes, membership training records) from the targeted estimator. This method
        should be overridden by all concrete inference attack implementations.

        :param x: An array with reference inputs to be used in the attack.
        :param y: Labels for `x`. This parameter is only used by some of the attacks.
        :return: An array holding the inferred properties.
        """
        raise NotImplementedError


class MembershipInferenceAttack(InferenceAttack):
    """
    Abstract base class for membership inference attack classes.
    """

    def __init__(self, estimator):
        """
        :param estimator: A trained estimator targeted for inference attack.
        :type estimator: :class:`.art.estimators.estimator.BaseEstimator`
        :param attack_feature: The index of the feature to be attacked.
        """
        super().__init__(estimator)

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer membership status of samples from the target estimator. This method
        should be overridden by all concrete inference attack implementations.
        :param x: An array with reference inputs to be used in the attack.
        :param y: Labels for `x`. This parameter is only used by some of the attacks.
        :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just
                              the predicted class.
        :return: An array holding the inferred membership status (1 indicates member of training set,
                 0 indicates non-member) or class probabilities.
        """
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.
        """
        # Save attack-specific parameters
        super().set_params(**kwargs)
        self._check_params()

class AttributeInferenceAttack(InferenceAttack):
    """
    Abstract base class for attribute inference attack classes.
    """

    attack_params = InferenceAttack.attack_params + ["attack_feature"]

    def __init__(self, estimator, attack_feature: Union[int, slice] = 0):
        """
        :param estimator: A trained estimator targeted for inference attack.
        :type estimator: :class:`.art.estimators.estimator.BaseEstimator`
        :param attack_feature: The index of the feature to be attacked.
        """
        super().__init__(estimator)
        self.attack_feature = attack_feature

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer sensitive properties (attributes, membership training records) from the targeted estimator. This method
        should be overridden by all concrete inference attack implementations.

        :param x: An array with reference inputs to be used in the attack.
        :param y: Labels for `x`. This parameter is only used by some of the attacks.
        :return: An array holding the inferred properties.
        """
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.
        """
        # Save attack-specific parameters
        super().set_params(**kwargs)
        self._check_params()


class ReconstructionAttack(Attack):
    """
    Abstract base class for reconstruction attack classes.
    """

    attack_params = InferenceAttack.attack_params

    def __init__(self, estimator):
        """
        :param estimator: A trained estimator targeted for reconstruction attack.
        """
        super().__init__(estimator)

    @abc.abstractmethod
    def reconstruct(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct the training dataset of and from the targeted estimator. This method
        should be overridden by all concrete inference attack implementations.

        :param x: An array with known records of the training set of `estimator`.
        :param y: An array with known labels of the training set of `estimator`, if None predicted labels will be used.
        :return: A tuple of two arrays for the reconstructed training input and labels.
        """
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.
        """
        # Save attack-specific parameters
        super().set_params(**kwargs)
        self._check_params()
