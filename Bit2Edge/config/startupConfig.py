# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This should be called ahead to speed up the program when train the TensorFlow model (if/elif using the GPU).
# References:
# - https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html?fbclid=IwAR3fjhy_H0BUC
# Aob-sSbaN_b6dyCHryT-L3k0_d9aHGPDi1EPgYOEATXcs0#tf_disable_cublas_tensor_op_math
# - https://stackoverflow.com/questions/58984892/how-to-set-environment-variable-tf-keras-1-for-onnx-conversion

import os
from logging import warning
from typing import Dict, Union, Tuple, Callable

COMPUTING_DEVICE: Dict[str, Union[bool, str]] = \
    {
        'activate': False, 'warning': False, 'Computing Device': 'GPU',
    }

# Preprocessing.ReadFile
EXTRA_LIBRARY: Dict[str, bool] = {'Dask': False, 'Dask_activated': False}
_INFO_LABELS_PREBUILT: Tuple[str, ...] = ('Molecule', 'Radical X', 'Radical Y', 'Bond Index', 'Bond Type')
GetPrebuiltInfoLabels: Callable = lambda: list(_INFO_LABELS_PREBUILT)


def _CastValue(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return 'true' if value is True else 'false'
    if isinstance(value, int):
        return str(value)
    raise ValueError


def SetFlag(key: str, value) -> None:
    value = _CastValue(value)
    os.environ[key] = value
    COMPUTING_DEVICE[key] = value


def OptimizeTensorFlow(device: int = 0, oneDNN: bool = False, cuBlasFp32: bool = True,
                       GpuMemGrowth: bool = False):
    if COMPUTING_DEVICE['activate'] is True:
        if COMPUTING_DEVICE['warning'] is False:
            warning('(One-Time Warning) This pipeline has been called at least once.')
            COMPUTING_DEVICE['warning']: bool = True
        return None

    SetFlag(key='CUDA_VISIBLE_DEVICES', value=int(device))
    SetFlag(key='TF_ENABLE_ONEDNN_OPTS', value=int(oneDNN))
    SetFlag(key='TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32', value=int(cuBlasFp32))  # If you have Tensor Core Math
    SetFlag(key='TF_FORCE_GPU_ALLOW_GROWTH', value=GpuMemGrowth)
    # SetFlag(key='TF_GPU_ALLOCATOR', value='cuda_malloc_async')

    if 'xla' in COMPUTING_DEVICE["Computing Device"].lower():
        SetFlag(key='TF_ENABLE_XLA', value='1')
        SetFlag(key='--xla_compile', value='1')
        SetFlag(key='TF_XLA_FLAGS', value='--tf_xla_enable_xla_devices --tf_xla_auto_jit=2')

    COMPUTING_DEVICE['activate'] = True


def EnableDevice():
    try:
        import tensorflow as tf
        import tensorflow_addons
    except (ImportError, ImportWarning):
        warning('TensorFlow is not available.')
        return None
    print('TF_VERSION: ', tf.__version__)
    print('TF_ADDONS_VERSION: ', tensorflow_addons.__version__)
    DEVICE: str = COMPUTING_DEVICE["Computing Device"].lower()
    try:
        if 'xla_gpu' in DEVICE:
            tf.device('/XLA_GPU:0')
        elif 'gpu' in DEVICE:
            tf.device('/GPU:0')
    except (ImportError, ImportWarning, RuntimeError):
        warning('TensorFlow is not installed or unable to optimize.')
    return None


RANDOM_SEED: Dict[str, Union[int, bool]] = \
    {
        # https://en.wikipedia.org/wiki/Mersenne_Twister
        'PYTHON_HASH_SEED_OS': 0, 'PYTHON_HASH_SEED_ACTIVATE': False,
        'PYTHON_RANDOM_MODULE': 0, 'PYTHON_MODULE_ACTIVATE': False,
        # numpy: https://www.cs.hmc.edu/tr/hmc-cs-2014-0905.pdf & https://www.pcg-random.org/
        'NUMPY_RANDOM': 0, 'NUMPY_ACTIVATE': False,

        # https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/random/set_seed
        'TENSORFLOW_RANDOM': 0, 'TENSORFLOW_ACTIVATE': False,
        'KERAS_UTILS_RANDOM_ALL': 0, 'KERAS_ACTIVATE': False,  # TF 2.7
    }


def ActivatePseudoRandomness() -> None:
    """ This function should not be activated """

    if RANDOM_SEED['PYTHON_HASH_SEED_ACTIVATE']:
        import os
        os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED['PYTHON_HASH_SEED_OS'])

    if RANDOM_SEED['PYTHON_MODULE_ACTIVATE']:
        import random
        random.seed(RANDOM_SEED['PYTHON_RANDOM_MODULE'])

    if RANDOM_SEED['KERAS_ACTIVATE']:
        import tensorflow
        major, minor, patch = [int(ver) for ver in tensorflow.__version__.split('.')]
        if major >= 2 and minor >= 7:
            try:
                from tensorflow.keras.utils import set_random_seed
                set_random_seed('KERAS_UTILS_RANDOM_ALL')
                return None
            except (ImportError, ImportWarning, AttributeError):
                from keras.utils import set_random_seed
                set_random_seed('KERAS_UTILS_RANDOM_ALL')
                warning('TensorFlow Library does not support this pipeline as it is available in TF 2.7+.')

    if RANDOM_SEED['NUMPY_ACTIVATE']:
        try:
            import numpy
            numpy.random.seed(RANDOM_SEED['NUMPY_RANDOM'])
        except (ImportError, ImportWarning):
            warning('Numpy-Random Library is not available.')

    if RANDOM_SEED['TENSORFLOW_ACTIVATE']:
        try:
            import tensorflow
            tensorflow.random.set_seed(RANDOM_SEED['TENSORFLOW_RANDOM'])
        except (ImportError, ImportWarning):
            warning('TensorFlow-Random Library is not available.')

    return None
