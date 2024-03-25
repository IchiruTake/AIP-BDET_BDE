# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import ndarray
from tensorflow import sparse
from tensorflow.keras.layers import Dense, BatchNormalization, Embedding
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import clone_model

from Bit2Edge.dataObject.DataBlock import DEFAULT_OUTPUT_NPDTYPE
from Bit2Edge.model.pool import ModelPool
from Bit2Edge.utils.file_io import FixPath, ExportFile, RemoveExtension
from Bit2Edge.utils.helper import InputFullCheck


def SwitchDatatype(dtype):
    np_dtype = (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64,
                np.float16, np.float32, np.float64)
    tf_dtype = (tf.uint8, tf.int8, tf.uint16, tf.int16, tf.uint32, tf.int32, tf.uint64, tf.int64,
                tf.float16, tf.float32, tf.float64)
    return tf_dtype[np_dtype.index(dtype)] if dtype in np_dtype else np_dtype[tf_dtype.index(dtype)]


def ConvertNumpyDenseToSparseTensor(data: ndarray) -> sparse.SparseTensor:
    return sparse.from_dense(data)


def EnforceToNumpyArray(arr: Union[ndarray, tf.Tensor]) -> ndarray:
    if isinstance(arr, ndarray):
        return arr
    try:
        return arr.numpy()
    except (ValueError, RuntimeError) as e:
        try:
            return np.array(arr, dtype=DEFAULT_OUTPUT_NPDTYPE)
        except Exception:
            raise e
    pass


def RemoveModelConfiguration(InputFile: str, OutputFile: Optional[str] = None) -> None:
    """
    This function will remove the gradient/optimizer from the saved model (configuration).

    Arguments:
    ---------

    InputFile: str
        The directory of your trained model that you want to modify.
    
    OutputFile: str
        The new directory of your model after filtering. If None, the new_directory is 
        equivalent as `{InputFile} - Output.h5`. Default to None.
    
    """
    # Hyper-parameter Verification
    InputFullCheck(InputFile, name='InputFile', dtype='str')
    InputFile = FixPath(FileName=InputFile, extension='.h5')
    if OutputFile is None:
        OutputFile = f'{RemoveExtension(InputFile, extension=".h5")} - Output.h5'
    else:
        InputFullCheck(OutputFile, name='OutputFile', dtype='str')
        OutputFile = FixPath(FileName=OutputFile, extension='.h5')

    current_model = load_model(InputFile, compile=False, custom_objects=None)
    try:
        new_model = clone_model(current_model)
        new_model.set_weights(weights=current_model.get_weights())
        new_model.save(OutputFile)
    except ValueError:
        current_model.save(OutputFile)


def GetModelWeights(InputFile: str, Folder: str = '') -> None:
    """
    This function will extract the learning weights in Dense, BatchNorm, and Embedding layer.

    Arguments:
    ---------

    InputFile: str
        The directory of your trained model that you want to modify.
    
    Folder: str
        The folder you want to store all the model parameter
    """
    # Hyper-parameter Verification
    if True:
        InputFullCheck(InputFile, name='InputFile', dtype='str')
        InputFile = FixPath(FileName=InputFile, extension='.h5')

        InputFullCheck(Folder, name='Folder', dtype='str')
        if len(Folder) != 0:
            Folder = FixPath(Folder, '/')

    # [1]: Initialization
    model: Model = load_model(InputFile, compile=False)
    layer_names: List[str] = [layer.name for layer in model.layers]
    counter: int = 0
    for model_index, name in enumerate(layer_names):
        # [2]: Retrieve Layer
        print(f'Layer #{model_index} - Name: {name}')
        layer = model.get_layer(name=name)
        if isinstance(layer, Dense):
            weight, bias = layer.get_weights()
            in_col: List[str] = [f'Input #{i}' for i in range(0, weight.shape[0])]
            in_col.append('Bias')
            out_col: List[str] = [f'Node #{i}' for i in range(0, weight.shape[1])]

            DataFrame = pd.DataFrame(data=np.concatenate((weight, np.atleast_2d(bias)), axis=0),
                                     index=in_col, columns=out_col)

        elif isinstance(layer, BatchNormalization):
            data = np.concatenate(layer.get_weights(), axis=0).reshape(4, -1)
            DataFrame = pd.DataFrame(data=data, index=['Gamma', 'Beta', 'Mean', 'Variance'],
                                     columns=[f'Feature #{temp}' for temp in range(0, data.shape[1])])
        elif isinstance(layer, Embedding):
            data = layer.get_weights()[0]
            DataFrame = pd.DataFrame(data=data, index=[f'Input #{i}' for i in range(0, data.shape[0])],
                                     columns=[f'Feature #{temp}' for temp in range(0, data.shape[1])])
        else:
            continue

        # [3]: Build File
        ExportFile(DataFrame=DataFrame, FilePath=f"{Folder}{name}-{counter}", index=True, )
        counter += 1
        del weight, bias

    return None


def CalcFlops(TF_Model: Model, batch_size: Optional[int] = None, specific: bool = False):
    # Link: https://github.com/tensorflow/tensorflow/issues/32809
    if batch_size is None:
        batch_size = 1
    else:
        InputFullCheck(batch_size, name='batch_size', dtype='int')
        batch_size = max(batch_size, 1)
    InputFullCheck(specific, name='specific', dtype='bool')

    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    print('TensorFlow:', tf.__version__)
    inputs = [tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in TF_Model.inputs]
    concrete_function = tf.function(TF_Model).get_concrete_function(inputs)
    frozen_func, graph_definition = convert_variables_to_constants_v2_as_graph(concrete_function)

    def calcFlops(specificMode: bool):
        v1 = tf.compat.v1
        cmd = 'op' if not specificMode else 'scope'
        run_meta = v1.RunMetadata()
        options = v1.profiler.ProfileOptionBuilder.float_operation()
        flops = v1.profiler.profile(graph=graph, run_meta=run_meta, cmd=cmd, options=options)
        return flops.total_float_ops

    if not specific:
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_definition, name='')
            return calcFlops(specificMode=specific)
    return calcFlops(specificMode=specific)


def LoadModel(filepath: Optional[str], compilation: bool = False) -> Optional[Model]:
    if filepath is None:
        return None

    InputFullCheck(filepath, name='TF_Model', dtype='str', delimiter='-')
    return ModelPool.GetModel(filepath=filepath, compilation=compilation)


def TryResetStateInTFModel(model: Optional[Model]) -> None:
    if model is not None:
        model.reset_metrics()
        model.reset_states()
    return None


def HasBatchNorm(model: Model) -> bool:
    return any(isinstance(layer, BatchNormalization) for layer in model.layers)
