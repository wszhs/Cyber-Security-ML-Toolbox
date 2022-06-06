from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from numpy.lib.function_base import copy
np.random.seed(0)
from tqdm.auto import trange
from csmt.config import ART_NUMPY_DTYPE
from csmt.attacks.attack import EvasionAttack
from csmt.estimators.estimator import BaseEstimator
from csmt.estimators.classification.classifier import ClassifierMixin
from csmt.utils import check_and_transform_label_format
import copy

class ZooAttack(EvasionAttack):

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: None,
        learning_rate: float = 1e-2,
        max_iter: int = 10,
        abort_early: bool = True,
        nb_parallel: int = 128,
        batch_size: int = 1,
        variable_h: float = 1e-4
    ):
        super().__init__(estimator=classifier)

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.abort_early = abort_early
        self.nb_parallel = nb_parallel
        self.batch_size = batch_size
        self.variable_h = variable_h
        self.verbose = True

        # Initialize some internal variables
        self._init_size = 32
        self.nb_parallel = nb_parallel
        print(self.estimator.input_shape)
        self._current_noise = np.zeros((batch_size,) + self.estimator.input_shape, dtype=ART_NUMPY_DTYPE)

        self.adam_mean = None
        self.adam_var = None
        self.adam_epochs = None

    def loss(
        self, x: np.ndarray, x_adv: np.ndarray, target: np.ndarray):
        l2dist = np.sum(np.square(x - x_adv), axis=1)
        preds = self.estimator.predict(np.array(x_adv), batch_size=self.batch_size)
        z_target = np.sum(preds * target, axis=1)
        z_other = np.max(preds * (1 - target) + (np.min(preds, axis=1) - 1)[:, np.newaxis] * target, axis=1,)
        loss = np.maximum(z_target - z_other, 0)
        c_weight=1
        return preds, l2dist, c_weight * loss+l2dist

    def _loss(
        self, x: np.ndarray, x_adv: np.ndarray, target: np.ndarray):
        l2dist = np.sum(np.square(x - x_adv), axis=1)
        preds = self.estimator.predict(np.array(x_adv), batch_size=self.batch_size)
        z_target = np.sum(preds * target, axis=1)
        z_other = np.max(preds * (1 - target) + (np.min(preds, axis=1) - 1)[:, np.newaxis] * target, axis=1,)
        loss = np.maximum(z_target - z_other, 0)
        c_weight=1
        return preds, l2dist, c_weight * loss+l2dist

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        #路径初始化为0
        X_adv_path=np.zeros((x.shape[0],2,x.shape[1]))
        orig_y=copy.deepcopy(y)
        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        # Check that `y` is provided for targeted attacks
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # Compute adversarial examples with implicit batching
        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))
        x_adv = []
        for batch_id in trange(nb_batches, desc="ZOO", disable=not self.verbose):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]
            res = self._generate_batch(x_batch, y_batch)
            x_adv.append(res)
        x_adv = np.vstack(x_adv)

        # Apply clip
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            np.clip(x_adv, clip_min, clip_max, out=x_adv)
        return x_adv,orig_y,X_adv_path

    def _generate_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:

        best_attack = self._generate_bss(x_batch, y_batch)

        return best_attack

    def _generate_bss(self, x_batch: np.ndarray, y_batch: np.ndarray):

        x_orig = x_batch.astype(ART_NUMPY_DTYPE)
        x_orig = x_batch
        self._reset_adam(np.prod(self.estimator.input_shape).item())
        if x_batch.shape == self._current_noise.shape:
            self._current_noise.fill(0)
        else:
            self._current_noise = np.zeros(x_batch.shape, dtype=ART_NUMPY_DTYPE)
        x_adv = x_orig.copy()

        # Initialize best distortions, best changed labels and best attacks
        dist_arr=np.ones((self.max_iter))*np.inf
        label_arr = np.ones((self.max_iter))
        attack_arr=np.repeat(x_adv,self.max_iter,axis=0)
        interaction_arr=np.ones((self.max_iter))*np.inf

        for iter_ in range(self.max_iter):

            # Compute adversarial examples and loss

            x_adv = self._optimizer(x_adv, y_batch)

            preds, l2dist, loss = self.loss(x_orig, x_adv, y_batch)


            dist_arr[iter_]=l2dist
            label_arr[iter_]=np.argmax(preds, axis=1)
            attack_arr[iter_]=x_adv

        final_loss=np.inf
        best_attack=attack_arr[0]
        for i in range(self.max_iter):
            cur_loss=dist_arr[i]
            if cur_loss<final_loss and label_arr[i]==0:
                best_attack=attack_arr[i]
                final_loss=cur_loss

        return best_attack

    def _optimizer(self, x: np.ndarray, targets: np.ndarray) -> np.ndarray:
        # Variation of input for computing loss, same as in original implementation
        coord_batch = np.repeat(self._current_noise, 2 * self.nb_parallel, axis=0)
        coord_batch = coord_batch.reshape(2 * self.nb_parallel * self._current_noise.shape[0], -1)

        # Sample indices to prioritize for optimization
        indices = (np.random.choice(coord_batch.shape[-1] * x.shape[0], self.nb_parallel * self._current_noise.shape[0], replace=False,)% coord_batch.shape[-1])

        # Create the batch of modifications to run
        for i in range(self.nb_parallel * self._current_noise.shape[0]):
            coord_batch[2 * i, indices[i]] += self.variable_h
            coord_batch[2 * i + 1, indices[i]] -= self.variable_h

        # Compute loss for all samples and coordinates, then optimize
        expanded_x = np.repeat(x, 2 * self.nb_parallel, axis=0).reshape((-1,) + x.shape[1:])
        expanded_targets = np.repeat(targets, 2 * self.nb_parallel, axis=0).reshape((-1,) + targets.shape[1:])
        _, _, loss = self._loss(
            expanded_x, expanded_x + coord_batch.reshape(expanded_x.shape), expanded_targets,
        )
        self._current_noise = self._optimizer_adam_coordinate(loss,indices,self.adam_mean,self.adam_var,self._current_noise,self.learning_rate,self.adam_epochs,True,)
        return x + self._current_noise

    def _optimizer_adam_coordinate(self,losses: np.ndarray,index: int,mean: np.ndarray,var: np.ndarray,current_noise: np.ndarray,learning_rate: float,adam_epochs: np.ndarray,proj: bool,) -> np.ndarray:

        beta1, beta2 = 0.9, 0.999

        # Estimate grads from loss variation (constant `h` from the paper is fixed to .0001)
        grads = np.array([(losses[i] - losses[i + 1]) / (2 * self.variable_h) for i in range(0, len(losses), 2)])

        # ADAM update
        mean[index] = beta1 * mean[index] + (1 - beta1) * grads
        var[index] = beta2 * var[index] + (1 - beta2) * grads ** 2

        corr = (np.sqrt(1 - np.power(beta2, adam_epochs[index]))) / (1 - np.power(beta1, adam_epochs[index]))
        orig_shape = current_noise.shape
        current_noise = current_noise.reshape(-1)
        current_noise[index] -= learning_rate * corr * mean[index] / (np.sqrt(var[index]) + 1e-8)
        adam_epochs[index] += 1

        return current_noise.reshape(orig_shape)

    def _optimizer_adam_coordinate_zhs(self,losses: np.ndarray,index: int,current_noise: np.ndarray,learning_rate: float) -> np.ndarray:

        mean = np.zeros(np.prod(self.estimator.input_shape).item(), dtype=ART_NUMPY_DTYPE)
        # Estimate grads from loss variation (constant `h` from the paper is fixed to .0001)
        grads = np.array([(losses[i] - losses[i + 1]) / (2 * self.variable_h) for i in range(0, len(losses), 2)])

        mean[index] = grads

        orig_shape = current_noise.shape
        current_noise = current_noise.reshape(-1)
        current_noise[index] -= learning_rate * mean[index]

        return current_noise.reshape(orig_shape)

    def _reset_adam(self, nb_vars: int, indices: Optional[np.ndarray] = None) -> None:
        # If variables are already there and at the right size, reset values
        if self.adam_mean is not None and self.adam_mean.size == nb_vars:
            if indices is None:
                self.adam_mean.fill(0)
                self.adam_var.fill(0)  # type: ignore
                self.adam_epochs.fill(1)  # type: ignore
            else:
                self.adam_mean[indices] = 0
                self.adam_var[indices] = 0  # type: ignore
                self.adam_epochs[indices] = 1  # type: ignore
        else:
            # Allocate Adam variables
            self.adam_mean = np.zeros(nb_vars, dtype=ART_NUMPY_DTYPE)
            self.adam_var = np.zeros(nb_vars, dtype=ART_NUMPY_DTYPE)
            self.adam_epochs = np.ones(nb_vars, dtype=np.int32)