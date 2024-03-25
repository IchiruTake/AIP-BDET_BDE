# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module serves to be the customized layer of the model. The customized layer is
# a just a collection of Dense, Activation, and Concat/Add layer together that are all
# implemented in Tensorflow.Keras. 
# --------------------------------------------------------------------------------

from logging import info
from functools import lru_cache
from math import sqrt, log2
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from numpy import ndarray
from tensorflow.keras.layers import Activation
# From v2.6, Tensorflow.Keras is a directory passed to Keras library.
from tensorflow.keras.layers import Concatenate, Dense, Input, Layer, Dropout

from Bit2Edge.config.modelConfig import MODEL_STRUCTURE
from Bit2Edge.utils.verify import (InputFullCheck, InputCheckRange, InputCheckIterable,
                                   TestState, TestStateByWarning, TestStateByInfo)

try:
    import tensorflow_addons.activations

    TF_ADDONS: bool = True
except (ImportError, ImportWarning) as e:
    TF_ADDONS: bool = False


def _AdjustNeurons_(neurons: Union[int, float], factor: Union[int, float], op: str) -> int:
    """
    This function would manipulate the number of neurons in a layer by a factor, either scaling
    up or down.

    Arguments:
    ---------

    neurons : int or float
        The number of neurons in the layer.
    
    factor : int or float
        The factor we want to scale up or down the number of neurons. Must be the absolute value
    
    op : str
        The operator we want to use. It can only be either '+' or '-'.

    """
    InputFullCheck(op, name='operator', dtype='str')
    TestState(op in ('-', '+'), 'The operator can only be either prune(-) and enlarge (+)')
    InputCheckRange(neurons, name='neurons', maxValue=None, minValue=0, allowFloatInput=True)
    InputCheckRange(factor, name='pruning_factor' if op == '-' else 'enlarge_factor',
                    maxValue=1 if op == '-' else None, minValue=0, allowNoneInput=False, allowFloatInput=True)
    if factor == 0:
        return int(neurons)
    return int(neurons * (1 - factor)) if op == '-' else int(neurons * (1 + factor))


def Prune(neurons: Union[int, float], factor: Union[int, float]) -> int:
    return _AdjustNeurons_(neurons=neurons, factor=factor, op='-')


def Enlarge(neurons: Union[int, float], factor: Union[int, float]) -> int:
    return _AdjustNeurons_(neurons=neurons, factor=factor, op='+')

@lru_cache(maxsize=16)
def _ScaleByTargetSize(TargetSize: int) -> float:
    return max(1.0, sqrt((TargetSize + 2) / (1 + 2)))


def GetIdealNeurons(FeatureDistributions: List, Indices: Union[List[int], Tuple[int, ...]],
                    TargetSize: int) -> List[int]:
    """
    This function is to calculate the number of neurons in each layer of the M-model, which combines the IEEE 2003 -
    Learning Capability and Storage Capacity of Two-Hidden-Layer Feedforward Networks, by Guang-Bin Huang with some
    modifications ahead to minimize connections. We also give credit to Katsunari Shibata, and Yusuke Ikeda (2009) -
    Effect of number of hidden neurons on learning in large-scale layered neural networks.

    Reference: https://doi.org/10.1109/TNN.2003.809401

    Arguments:
    ---------

    FeatureDistributions : List
        The list of feature distribution of each layer.

    Indices : List[int] or Tuple[int, ...]
        The list of indices of the feature distribution we want to use in our model.
    
    TargetSize : int
        The number of target functions we want to train our model.
    
    hidden : int
        The number of hidden layers in Guang Bin-Huang's architecture, which would be used for our M-model.
        Default to 3. The original paper uses 2 hidden layers, but we want to make it more flexible.
    
    PruneRatio : List[float]
        Optional argument. It calculated the prune ratio we want to apply in our architecture.
    
    Returns:
    -------
    
    A list of neurons represented by integer but is sorted reversely
    
    """
    # Hyper-parameter Verification
    InputCheckRange(TargetSize, name='TargetSize', maxValue=None, minValue=1, allowFloatInput=False)
    TestStateByInfo(TargetSize <= 1, f'There are {TargetSize} targets in training.')

    results: List[int] = []
    size: int = sum(FeatureDistributions[idx]['size'] for idx in Indices)
    n: int = FeatureDistributions[0]['n']

    print("Computing the number of neurons in each layer...")
    l1: int = sum(FeatureDistributions[idx][MODEL_STRUCTURE.get('M-model Type', '99r')] for idx in Indices)
    l1 += sqrt(size * log2(n))
    l1 = int(l1 * _ScaleByTargetSize(TargetSize))
    results.append(l1)
    print(f'Number of neurons in the first layer: {l1}')

    l2: int = int(TargetSize * sqrt(n / (TargetSize + 2)) + sqrt(size * log2(n))) // 2
    l3: int = int(sqrt(TargetSize * size))
    l2 = l2 - l3 // int(4 / TargetSize)
    print(f'Number of neurons in the second layer: {l2}')
    results.append(l2)
    print(f'Number of neurons in the third layer: {l3}')
    results.append(l3)

    results.sort(reverse=True)
    return results


# ----------------------------------------------------------------------------------------------------------
def _Connect_(fromLayer: Layer, toLayer: Layer, initMode: bool) -> Layer:
    return toLayer if initMode else toLayer(fromLayer)


def ComputeSizeFromCustomLayer(units: Union[int, float], TargetOffset: int = 1) -> int:
    return int(int(units) * (TargetOffset ** 0.5))


def CustomLayer(layer: Layer, units: Optional[Union[int, float]] = None,
                activation: Optional[str] = 'relu', TargetOffset: int = 1,
                initMode: bool = False, dense: bool = True, bias: bool = MODEL_STRUCTURE['use_bias'],
                dynamic: bool = False, name: Optional[str] = None,
                dtype: Union[np.dtype, tf.DType, str] = 'float32',
                dropout: Union[int, float] = 0) -> Layer:
    """
    This function is to standardize the creation and connection of the Dense layer.

    Arguments:
    ---------

    layer : Layer
        The previous tensorflow layer.

    units : int or float
        The **base** number of neuron units for this Dense layer.
        Scaled by the built-in `int()` function.
    
    activation : str
        The activation function of this Dense layer. Default to 'relu'.
    
    TargetOffset : int
        The number of target for prediction. This would scale the :arg:`units` by 
        sqrt(:arg:`TargetOffset`) times, which is computed by :meth:`ComputeSizeFromCustomLayer()`.
        Default to 1 (not scaled).
    
    initMode : bool
        If True, the layer denys the connection to the previous layer (:arg:`layer`). 
        Only the objects is created. Default to False.
    
    dense : bool
        If True, the output is the TF Dense-layer. Otherwise, this is just an activation layer only.
        Default to True.
    
    bias : bool
        Whether the Dense layer used the bias for computation. Default to 
        config.MODEL_STRUCTURE['use_bias'] (True).
    
    dynamic : bool
        A TensorFlow flag to determine whether the Dense layer is dynamic. Default to False.
    
    name : str
        The output layer's name of this layer. Default to None which is determined automatically
        by TensorFlow.
    
    dtype : np.dtype or tf.DType or str
        The data-type of the weights of the output layer. Default to 'float32'.
    
    dropout : float
        The dropout rate of this layer. Default to 0 (no dropout). This Dropout is connected
        to the previous layer (:arg:`layer`) and this layer. The Dropout is not used if
        :arg:`dense` is False.
    
    Returns:
    -------

    A tensorflow layer object.

    """

    if activation is not None:
        if activation.find('Addons') == -1:
            activation = activation.lower()
    else:
        TestState(dense, 'In this model, we disable the creation of one Linear layer only.')
    if units == TargetOffset and TargetOffset != 1:
        info('This layer could be the final layer in the model, '
             'but the offset scaling is not equal to one.')
    if dense and isinstance(units, float):
        units: int = int(units)
    units = ComputeSizeFromCustomLayer(units=units, TargetOffset=TargetOffset)
    try:
        if dense:
            InputCheckRange(dropout, name='dropout', maxValue=1, minValue=0, allowFloatInput=True,
                            rightBound=True)
            if dropout != 0:
                TestStateByWarning(dropout < 0.20, 'The proportion of units to be dropped is overwhelming.')
                layer = Dropout(rate=dropout, noise_shape=None, seed=None)(layer)

            new_layer = Dense(units, activation=activation, use_bias=bias,
                              dynamic=dynamic, dtype=dtype, name=name)
        else:
            new_layer = Activation(activation, dtype=dtype, name=name)
        return _Connect_(fromLayer=layer, toLayer=new_layer, initMode=initMode)

    except ValueError as e1:
        if not TF_ADDONS or activation.find('Addons') != -1:
            raise e1

    return CustomLayer(layer, units=units, TargetOffset=TargetOffset, activation=f'Addons>{activation}',
                       initMode=initMode, dense=dense, bias=bias,
                       dynamic=dynamic, name=name, dtype=dtype, dropout=dropout)


def CustomInput(size: int, name: Optional[str] = None, sparseState: Optional[bool] = None) -> Input:
    if not isinstance(size, int):
        size: int = int(size)
    if name is not None:    # Replace spacing and dash
        name: str = name.strip().replace(' ', '_').replace('-', '_').lower()
    return Input(shape=(size,), batch_size=None, name=name, dtype='uint8', sparse=sparseState)


# ----------------------------------------------------------------------------------------------------------
def JoinLayer(layers: List[Layer], TargetSize: int, name: Optional[str] = None) -> Layer:
    return CustomLayer(Concatenate()(layers), TargetSize, TargetOffset=1, name=name,
                       activation=MODEL_STRUCTURE['Output Last-Act'], bias=MODEL_STRUCTURE['use_bias'])
