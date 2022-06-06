'''
Author: your name
Date: 2021-04-01 15:27:11
LastEditTime: 2021-07-13 12:22:59
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/config.py
'''

ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------- CONSTANTS AND TYPES

ART_NUMPY_DTYPE = np.float64
ART_DATA_PATH: str

# --------------------------------------------------------------------------------------------- DEFAULT PACKAGE CONFIGS

_folder = os.path.expanduser("~")
if not os.access(_folder, os.W_OK):
    _folder = "/tmp"
_folder = os.path.join(_folder, ".art")


def set_data_path(path):
    """
    Set the path for ART's data directory (ART_DATA_PATH).
    """
    expanded_path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(expanded_path, exist_ok=True)
    if not os.access(expanded_path, os.R_OK):
        raise OSError(f"path {expanded_path} cannot be read from")
    if not os.access(expanded_path, os.W_OK):
        logger.warning(f"path %s is read only", expanded_path)

    global ART_DATA_PATH
    ART_DATA_PATH = expanded_path
    logger.info(f"set ART_DATA_PATH to %s", expanded_path)


# Load data from configuration file if it exists. Otherwise create one.
_config_path = os.path.expanduser(os.path.join(_folder, "config.json"))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)

            # Since renaming this variable we must update existing config files
            if "DATA_PATH" in _config:
                _config["ART_DATA_PATH"] = _config.pop("DATA_PATH")
                try:
                    with open(_config_path, "w") as f:
                        f.write(json.dumps(_config, indent=4))
                except IOError:
                    logger.warning("Unable to update configuration file", exc_info=True)

    except ValueError:
        _config = {}

if not os.path.exists(_folder):
    try:
        os.makedirs(_folder)
    except OSError:
        logger.warning("Unable to create folder for configuration file.", exc_info=True)

if not os.path.exists(_config_path):
    # Generate default config
    _config = {"ART_DATA_PATH": os.path.join(_folder, "data")}

    try:
        with open(_config_path, "w") as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        logger.warning("Unable to create configuration file", exc_info=True)

if "ART_DATA_PATH" in _config:
    set_data_path(_config["ART_DATA_PATH"])