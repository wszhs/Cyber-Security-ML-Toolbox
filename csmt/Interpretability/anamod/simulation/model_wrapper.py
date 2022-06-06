"""Generates model for simulation - perturbed version of ground truth polynomial"""


class ModelWrapper():
    """Class implementing model API required by anamod"""
    def __init__(self, ground_truth_model, noise_multiplier):
        self.ground_truth_model = ground_truth_model
        self.noise_multiplier = noise_multiplier

    # pylint: disable = invalid-name
    def predict(self, X):
        """Perform prediction on input X (comprising one or more instances)"""
        return self.ground_truth_model.predict(X, noise=self.noise_multiplier)
