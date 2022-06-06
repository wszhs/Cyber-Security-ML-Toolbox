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
This module implements the classifier `KerasClassifier` for Keras models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import numpy as np
import six

from csmt import config
from csmt.estimators.keras import KerasEstimator
from csmt.estimators.classification.classifier import (
    ClassifierMixin,
    ClassGradientsMixin,
)
from csmt.utils import Deprecated, deprecated_keyword_arg, check_and_transform_label_format

logger = logging.getLogger(__name__)

KERAS_MODEL_TYPE = Union["keras.models.Model", "tf.keras.models.Model"]


class KerasClassifier(ClassGradientsMixin, ClassifierMixin, KerasEstimator):
    """
    Wrapper class for importing Keras models.
    """

    @deprecated_keyword_arg("channel_index", end_version="1.6.0", replaced_by="channels_first")
    def __init__(
        self,
        model: KERAS_MODEL_TYPE,
        use_logits: bool = False,
        channel_index=Deprecated,
        channels_first: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        input_layer: int = 0,
        output_layer: int = 0,
    ) -> None:
        """
        Create a `Classifier` instance from a Keras model. Assumes the `model` passed as argument is compiled.

        :param model: Keras model, neural network or other.
        :param use_logits: True if the output of the model are logits; false for probabilities or any other type of
               outputs. Logits output should be favored when possible to ensure attack efficiency.
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param input_layer: The index of the layer to consider as input for models with multiple input layers. The layer
                            with this index will be considered for computing gradients. For models with only one input
                            layer this values is not required.
        :param output_layer: Which layer to consider as the output when the models has multiple output layers. The layer
                             with this index will be considered for computing gradients. For models with only one output
                             layer this values is not required.
        """
        # Remove in 1.6.0
        if channel_index == 3:
            channels_first = False
        elif channel_index == 1:
            channels_first = True
        elif channel_index is not Deprecated:
            raise ValueError("Not a proper channel_index. Use channels_first.")

        super().__init__(
            model=model,
            clip_values=clip_values,
            channel_index=channel_index,
            channels_first=channels_first,
        )

        self._input_layer = input_layer
        self._output_layer = output_layer

        if "<class 'tensorflow" in str(type(model).__mro__):
            self.is_tensorflow = True
        elif "<class 'keras" in str(type(model).__mro__):
            self.is_tensorflow = False
        else:
            raise TypeError("Type of model not recognized:" + str(type(model)))

        self._initialize_params(model, use_logits, input_layer, output_layer)

    def _initialize_params(
        self, model: KERAS_MODEL_TYPE, use_logits: bool, input_layer: int, output_layer: int,
    ):
        """
        Initialize most parameters of the classifier. This is a convenience function called by `__init__` and
        `__setstate__` to avoid code duplication.

        :param model: Keras model
        :param use_logits: True if the output of the model are logits.
        :param input_layer: Which layer to consider as the Input when the model has multiple input layers.
        :param output_layer: Which layer to consider as the Output when the model has multiple output layers.
        """
        # pylint: disable=E0401
        if self.is_tensorflow:
            import tensorflow as tf  # lgtm [py/repeated-import]

            if tf.executing_eagerly():
                raise ValueError("TensorFlow is executing eagerly. Please disable eager execution.")
            import tensorflow.keras as keras
            import tensorflow.keras.backend as k

            self._losses = keras.losses
        else:
            import keras  # lgtm [py/repeated-import]
            import keras.backend as k

            if hasattr(keras.utils, "losses_utils"):
                self._losses = keras.utils.losses_utils
            else:
                self._losses = None

        if hasattr(model, "inputs"):
            self._input_layer = input_layer
            self._input = model.inputs[input_layer]
        else:
            self._input = model.input
            self._input_layer = 0

        if hasattr(model, "outputs"):
            self._output = model.outputs[output_layer]
            self._output_layer = output_layer
        else:
            self._output = model.output
            self._output_layer = 0

        _, self._nb_classes = k.int_shape(self._output)
        self._input_shape = k.int_shape(self._input)[1:]
        logger.debug(
            "Inferred %i classes and %s as input shape for Keras classifier.", self.nb_classes, str(self.input_shape),
        )

        self._use_logits = use_logits

        # Get loss function
        if not hasattr(self._model, "loss"):
            logger.warning("Keras model has no loss set. Classifier tries to use `k.sparse_categorical_crossentropy`.")
            loss_function = k.sparse_categorical_crossentropy
        else:
            self._orig_loss = self._model.loss
            if isinstance(self._model.loss, six.string_types):
                loss_function = getattr(k, self._model.loss)

            elif "__name__" in dir(self._model.loss) and self._model.loss.__name__ in [
                "categorical_hinge",
                "categorical_crossentropy",
                "sparse_categorical_crossentropy",
                "binary_crossentropy",
                "kullback_leibler_divergence",
            ]:
                if self._model.loss.__name__ in [
                    "categorical_hinge",
                    "kullback_leibler_divergence",
                ]:
                    loss_function = getattr(keras.losses, self._model.loss.__name__)
                else:
                    loss_function = getattr(keras.backend, self._model.loss.__name__)

            elif isinstance(
                self._model.loss,
                (
                    keras.losses.CategoricalHinge,
                    keras.losses.CategoricalCrossentropy,
                    keras.losses.SparseCategoricalCrossentropy,
                    keras.losses.BinaryCrossentropy,
                    keras.losses.KLDivergence,
                ),
            ):
                loss_function = self._model.loss
            else:
                loss_function = getattr(k, self._model.loss.__name__)

        # Check if loss function is an instance of loss function generator, the try is required because some of the
        # modules are not available in older Keras versions
        try:
            flag_is_instance = isinstance(
                loss_function,
                (
                    keras.losses.CategoricalHinge,
                    keras.losses.CategoricalCrossentropy,
                    keras.losses.BinaryCrossentropy,
                    keras.losses.KLDivergence,
                ),
            )
        except AttributeError:
            flag_is_instance = False

        # Check if the labels have to be reduced to index labels and create placeholder for labels
        if (
            "__name__" in dir(loss_function)
            and loss_function.__name__
            in ["categorical_hinge", "categorical_crossentropy", "binary_crossentropy", "kullback_leibler_divergence",]
        ) or flag_is_instance:
            self._reduce_labels = False
            label_ph = k.placeholder(shape=self._output.shape)
        elif (
            "__name__" in dir(loss_function) and loss_function.__name__ in ["sparse_categorical_crossentropy"]
        ) or isinstance(loss_function, keras.losses.SparseCategoricalCrossentropy):
            self._reduce_labels = True
            label_ph = k.placeholder(shape=[None,])
        else:
            raise ValueError("Loss function not recognised.")

        # Define the loss using the loss function
        if "__name__" in dir(loss_function,) and loss_function.__name__ in [
            "categorical_crossentropy",
            "sparse_categorical_crossentropy",
            "binary_crossentropy",
        ]:
            loss_ = loss_function(label_ph, self._output, from_logits=self._use_logits)

        elif "__name__" in dir(loss_function) and loss_function.__name__ in [
            "categorical_hinge",
            "kullback_leibler_divergence",
        ]:
            loss_ = loss_function(label_ph, self._output)

        elif isinstance(
            loss_function,
            (
                keras.losses.CategoricalHinge,
                keras.losses.CategoricalCrossentropy,
                keras.losses.SparseCategoricalCrossentropy,
                keras.losses.KLDivergence,
                keras.losses.BinaryCrossentropy,
            ),
        ):
            loss_ = loss_function(label_ph, self._output)

        # Define loss gradients
        loss_gradients = k.gradients(loss_, self._input)

        if k.backend() == "tensorflow":
            loss_gradients = loss_gradients[0]
        elif k.backend() == "cntk":
            raise NotImplementedError("Only TensorFlow is supported as backend for Keras.")

        # Set loss, gradients and prediction functions
        self._predictions_op = self._output
        self._loss_function = loss_function
        self._loss = loss_
        self._loss_gradients = k.function([self._input, label_ph], [loss_gradients])

        # Get the internal layer
        self._layer_names = self._get_layers()

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    def loss(self, x: np.ndarray, y: np.ndarray, reduction: str = "none", **kwargs) -> np.ndarray:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                   'none': no reduction will be applied
                   'mean': the sum of the output will be divided by the number of elements in the output,
                   'sum': the output will be summed.
        :return: Loss values.
        :rtype: Format as expected by the `model`
        """
        if not self._losses:
            raise NotImplementedError("loss method is only supported for keras versions >= 2.3.1")

        if self.is_tensorflow:
            import tensorflow.keras.backend as k
        else:
            import keras.backend as k

        x_preprocessed, y_preprocessed = x,y
        shape_match = [i is None or i == j for i, j in zip(self._input_shape, x_preprocessed.shape[1:])]
        if not all(shape_match):
            raise ValueError(
                "Error when checking x: expected preprocessed x to have shape {} but got array with "
                "shape {}.".format(self._input_shape, x_preprocessed.shape[1:])
            )

        # Adjust the shape of y for loss functions that do not take labels in one-hot encoding
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        predictions = self._model.predict(x_preprocessed)

        if self._orig_loss and hasattr(self._orig_loss, "reduction"):
            prev_reduction = self._orig_loss.reduction
            self._orig_loss.reduction = self._losses.Reduction.NONE
            loss = self._orig_loss(y_preprocessed, predictions)
            self._orig_loss.reduction = prev_reduction
        else:
            prev_reduction = []
            predictions = k.constant(predictions)
            y_preprocessed = k.constant(y_preprocessed)
            for loss_function in self._model.loss_functions:
                prev_reduction.append(loss_function.reduction)
                loss_function.reduction = self._losses.Reduction.NONE
            loss = self._loss_function(y_preprocessed, predictions)
            for i, loss_function in enumerate(self._model.loss_functions):
                loss_function.reduction = prev_reduction[i]

        loss_value = k.eval(loss)

        if reduction == "none":
            pass
        elif reduction == "mean":
            loss_value = np.mean(loss_value, axis=0)
        elif reduction == "sum":
            loss_value = np.sum(loss_value, axis=0)

        return loss_value

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: Array of gradients of the same shape as `x`.
        """
        # Check shape of preprocessed `x` because of custom function for `_loss_gradients`
        x_preprocessed, y_preprocessed = x,y
        shape_match = [i is None or i == j for i, j in zip(self._input_shape, x_preprocessed.shape[1:])]
        if not all(shape_match):
            raise ValueError(
                "Error when checking x: expected preprocessed x to have shape {} but got array with shape {}".format(
                    self._input_shape, x_preprocessed.shape[1:]
                )
            )

        # Adjust the shape of y for loss functions that do not take labels in one-hot encoding
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        # Compute gradients
        gradients = self._loss_gradients([x_preprocessed, y_preprocessed])[0]

        return gradients

    def class_gradient(self, x: np.ndarray, label: Optional[Union[int, List[int]]] = None, **kwargs) -> np.ndarray:
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values are provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        # Check value of label for computing gradients
        if not (
            label is None
            or (isinstance(label, (int, np.integer)) and label in range(self.nb_classes))
            or (
                isinstance(label, np.ndarray)
                and len(label.shape) == 1
                and (label < self.nb_classes).all()
                and label.shape[0] == x.shape[0]
            )
        ):
            raise ValueError("Label %s is out of range." % str(label))

        # Check shape of preprocessed `x` because of custom function for `_class_gradients`
        x_preprocessed= x
        shape_match = [i is None or i == j for i, j in zip(self._input_shape, x_preprocessed.shape[1:])]
        if not all(shape_match):
            raise ValueError(
                "Error when checking x: expected preprocessed x to have shape {} but got array with shape {}".format(
                    self._input_shape, x_preprocessed.shape[1:]
                )
            )

        self._init_class_gradients(label=label)

        if label is None:
            # Compute the gradients w.r.t. all classes
            gradients = np.swapaxes(np.array(self._class_gradients([x_preprocessed])), 0, 1)

        elif isinstance(label, (int, np.integer)):
            # Compute the gradients only w.r.t. the provided label
            gradients = np.swapaxes(np.array(self._class_gradients_idx[label]([x_preprocessed])), 0, 1)  # type: ignore
            assert gradients.shape == (x_preprocessed.shape[0], 1) + x_preprocessed.shape[1:]

        else:
            # For each sample, compute the gradients w.r.t. the indicated target class (possibly distinct)
            unique_label = list(np.unique(label))
            gradients = np.array([self._class_gradients_idx[l]([x_preprocessed]) for l in unique_label])
            gradients = np.swapaxes(np.squeeze(gradients, axis=1), 0, 1)
            lst = [unique_label.index(i) for i in label]
            gradients = np.expand_dims(gradients[np.arange(len(gradients)), lst], axis=1)

        return gradients

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :param batch_size: Size of batches.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        from csmt.config import ART_NUMPY_DTYPE

        # Apply preprocessing
        # 当模型是序列模型时
        # x_preprocessed= np.reshape(x,(x.shape[0],x.shape[1],1))
        x_preprocessed= x

        # Run predictions with batching
        predictions = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=ART_NUMPY_DTYPE)
        for batch_index in range(int(np.ceil(x_preprocessed.shape[0] / float(batch_size)))):
            begin, end = (
                batch_index * batch_size,
                min((batch_index + 1) * batch_size, x_preprocessed.shape[0]),
            )
            predictions[begin:end] = self._model.predict([x_preprocessed[begin:end]])

        return predictions

    def fit(self, x: np.ndarray, y: np.ndarray, X_val,y_val,batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit_generator` function in Keras and will be passed to this function as such. Including the number of
               epochs or the number of steps per epoch as part of this argument will result in as error.
        """
        y = check_and_transform_label_format(y, self.nb_classes)

        # Apply preprocessing
        
        # 当模型是序列模型时
        # x_preprocessed, y_preprocessed = np.reshape(x,(x.shape[0],x.shape[1],1)),y
        x_preprocessed, y_preprocessed = x,y

        # Adjust the shape of y for loss functions that do not take labels in one-hot encoding
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)
        
        print(x_preprocessed.shape)
        self._model.fit(x_preprocessed, y_preprocessed,epochs=nb_epochs,batch_size=batch_size,validation_split=0.2)

        # gen = generator_fit(x_preprocessed, y_preprocessed, batch_size)
        # steps_per_epoch = max(int(x_preprocessed.shape[0] / batch_size), 1)
        # self._model.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=nb_epochs, **kwargs)

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :param layer: Layer for computing the activations.
        :param batch_size: Size of batches.
        :param framework: If true, return the intermediate tensor representation of the activation.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        # pylint: disable=E0401
        if self.is_tensorflow:
            import tensorflow.keras.backend as k
        else:
            import keras.backend as k
        from csmt.config import ART_NUMPY_DTYPE

        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError("Layer name %s is not part of the graph." % layer)
            layer_name = layer
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(self._layer_names):
                raise ValueError(
                    "Layer index %d is outside of range (0 to %d included)." % (layer, len(self._layer_names) - 1)
                )
            layer_name = self._layer_names[layer]
        else:
            raise TypeError("Layer must be of type `str` or `int`.")

        if x.shape == self.input_shape:
            x_expanded = np.expand_dims(x, 0)
        else:
            x_expanded = x

        # Apply preprocessing
        x_preprocessed= x

        if not hasattr(self, "_activations_func"):
            self._activations_func: Dict[str, Callable] = {}

        keras_layer = self._model.get_layer(layer_name)
        if layer_name not in self._activations_func:
            num_inbound_nodes = len(getattr(keras_layer, "_inbound_nodes", []))
            if num_inbound_nodes > 1:
                layer_output = keras_layer.get_output_at(0)
            else:
                layer_output = keras_layer.output
            self._activations_func[layer_name] = k.function([self._input], [layer_output])

        # Determine shape of expected output and prepare array
        output_shape = self._activations_func[layer_name]([x_preprocessed[0][None, ...]])[0].shape
        activations = np.zeros((x_preprocessed.shape[0],) + output_shape[1:], dtype=ART_NUMPY_DTYPE)

        # Get activations with batching
        for batch_index in range(int(np.ceil(x_preprocessed.shape[0] / float(batch_size)))):
            begin, end = (
                batch_index * batch_size,
                min((batch_index + 1) * batch_size, x_preprocessed.shape[0]),
            )
            activations[begin:end] = self._activations_func[layer_name]([x_preprocessed[begin:end]])[0]

        if framework:
            placeholder = k.placeholder(shape=x.shape)
            return placeholder, keras_layer(placeholder)
        else:
            return activations

    def custom_loss_gradient(self, nn_function, tensors, input_values, name="default"):
        """
        Returns the gradient of the nn_function with respect to model input

        :param nn_function: an intermediate tensor representation of the function to differentiate
        :type nn_function: a Keras tensor
        :param tensors: the tensors or variables to differentiate with respect to
        :type tensors: `list`
        :param input_values: the inputs to evaluate the gradient
        :type input_values: `list`
        :param name: The name of the function. Functions of the same name are cached
        :type name: `str`
        :return: the gradient of the function w.r.t vars
        :rtype: `np.ndarray`
        """
        if self.is_tensorflow:
            import tensorflow.keras.backend as k
        else:
            import keras.backend as k

        if not hasattr(self, "_custom_loss_func"):
            self._custom_loss_func = {}

        if name not in self._custom_loss_func:
            grads = k.gradients(nn_function, tensors[0])[0]
            self._custom_loss_func[name] = k.function(tensors, [grads])

        outputs = self._custom_loss_func[name]
        return outputs(input_values)

    def _init_class_gradients(self, label: Optional[Union[int, List[int], np.ndarray]] = None) -> None:
        # pylint: disable=E0401
        if self.is_tensorflow:
            import tensorflow.keras.backend as k
        else:
            import keras.backend as k

        if len(self._output.shape) == 2:
            nb_outputs = self._output.shape[1]
        else:
            raise ValueError("Unexpected output shape for classification in Keras model.")

        if label is None:
            logger.debug("Computing class gradients for all %i classes.", self.nb_classes)
            if not hasattr(self, "_class_gradients"):
                class_gradients = [k.gradients(self._predictions_op[:, i], self._input)[0] for i in range(nb_outputs)]
                self._class_gradients = k.function([self._input], class_gradients)

        else:
            if isinstance(label, int):
                unique_labels = [label]
            else:
                unique_labels = np.unique(label)
            logger.debug("Computing class gradients for classes %s.", str(unique_labels))

            if not hasattr(self, "_class_gradients_idx"):
                self._class_gradients_idx = [None for _ in range(nb_outputs)]

            for current_label in unique_labels:
                if self._class_gradients_idx[current_label] is None:
                    class_gradients = [k.gradients(self._predictions_op[:, current_label], self._input)[0]]
                    self._class_gradients_idx[current_label] = k.function([self._input], class_gradients)

    def _get_layers(self) -> List[str]:
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        """
        # pylint: disable=E0401
        if self.is_tensorflow:
            from tensorflow.keras.layers import InputLayer
        else:
            from keras.engine.topology import InputLayer

        layer_names = [layer.name for layer in self._model.layers[:-1] if not isinstance(layer, InputLayer)]
        logger.info("Inferred %i hidden layers on Keras classifier.", len(layer_names))

        return layer_names

    def set_learning_phase(self, train: bool) -> None:
        """
        Set the learning phase for the backend framework.

        :param train: True to set the learning phase to training, False to set it to prediction.
        """
        # pylint: disable=E0401
        if self.is_tensorflow:
            import tensorflow.keras.backend as k
        else:
            import keras.backend as k

        if isinstance(train, bool):
            self._learning_phase = train
            k.set_learning_phase(int(train))

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file in the format specific to the backend framework. For Keras, .h5 format is used.

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

        self._model.save(str(full_path))
        logger.info("Model saved in path: %s.", full_path)

    def __getstate__(self) -> Dict[str, Any]:
        """
        Use to ensure `KerasClassifier` can be pickled.

        :return: State dictionary with instance parameters.
        """
        state = self.__dict__.copy()

        # Remove the unpicklable entries
        del state["_model"]
        del state["_input"]
        del state["_output"]
        del state["_predictions_op"]
        del state["_loss"]
        del state["_loss_gradients"]
        del state["_layer_names"]
        del state["_losses"]
        del state["_loss_function"]

        if "_orig_loss" in state:
            del state["_orig_loss"]

        if "_class_gradients" in state:
            del state["_class_gradients"]

        if "_class_gradients_idx" in state:
            del state["_class_gradients_idx"]

        if "_activations_func" in state:
            del state["_activations_func"]

        if "_custom_loss_func" in state:
            del state["_custom_loss_func"]

        model_name = str(time.time()) + ".h5"
        state["model_name"] = model_name
        self.save(model_name)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Use to ensure `KerasClassifier` can be unpickled.

        :param state: State dictionary with instance parameters to restore.
        """
        self.__dict__.update(state)

        if self.is_tensorflow:
            from tensorflow.keras.models import load_model
        else:
            from keras.models import load_model

        full_path = os.path.join(config.ART_DATA_PATH, state["model_name"])
        model = load_model(str(full_path))

        self._model = model
        self._initialize_params(model, state["_use_logits"], state["_input_layer"], state["_output_layer"])

    def __repr__(self):
        repr_ = (
            "%s(model=%r, use_logits=%r, channel_index=%r, channels_first=%r, clip_values=%r, input_layer=%r, output_layer=%r)"
            % (
                self.__module__ + "." + self.__class__.__name__,
                self._model,
                self._use_logits,
                self.channel_index,
                self.channels_first,
                self.clip_values,
                self._input_layer,
                self._output_layer,
            )
        )

        return repr_


def generator_fit(x: np.ndarray, y: np.ndarray, batch_size: int = 128) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Minimal data generator for randomly batching large datasets.

    :param x: The data sample to batch.
    :param y: The labels for `x`. The first dimension has to match the first dimension of `x`.
    :param batch_size: The size of the batches to produce.
    :return: A batch of size `batch_size` of random samples from `(x, y)`.
    """
    while True:
        indices = np.random.randint(x.shape[0], size=batch_size)
        yield x[indices], y[indices]
