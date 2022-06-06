# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import sys
from interpret.ext.extension_utils import load_class_extensions
from interpret.ext.extension import GREYBOX_EXTENSION_KEY, _is_valid_greybox_explainer

load_class_extensions(
    sys.modules[__name__], GREYBOX_EXTENSION_KEY, _is_valid_greybox_explainer
)
