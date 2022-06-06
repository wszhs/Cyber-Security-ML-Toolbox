"""Script that provides load/save functions for model"""

import pickle
import cloudpickle


def load_model(model_filename):
    """Load model from file"""
    with open(model_filename, "rb") as model_file:
        model = cloudpickle.load(model_file)
        return model


def save_model(model, model_filename):
    """Save model to file"""
    with open(model_filename, "wb") as model_file:
        cloudpickle.dump(model, model_file, protocol=pickle.DEFAULT_PROTOCOL)
