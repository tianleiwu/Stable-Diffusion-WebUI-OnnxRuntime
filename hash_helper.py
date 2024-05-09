# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import hashlib

import numpy as np


def hash_as_fp16_numpy_array(w: np.array):
    h = hashlib.blake2b(w.astype(np.float16).data.tobytes(), digest_size=20)
    return h.hexdigest()
