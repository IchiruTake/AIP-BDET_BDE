# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module serves as the model builder for the model. 
# --------------------------------------------------------------------------------

from logging import warning
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from tensorflow import sparse, convert_to_tensor
# From v2.6, Tensorflow.Keras is a directory passed to Keras library.
from tensorflow.keras.layers import Add, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras.engine.base_layer import Layer

from Bit2Edge.config.modelConfig import FRAMEWORK as framework
from Bit2Edge.config.modelConfig import MODEL_STRUCTURE as STRUCTURE
from Bit2Edge.dataObject.DataBlock import DEFAULT_OUTPUT_NPDTYPE
from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.dataObject.DatasetLinker import DatasetLinker
from Bit2Edge.dataObject.InputState import InputState
from Bit2Edge.model.TrainModelParams import TrainModelParams
from Bit2Edge.model.data import BE_2Input, BE_3Input, GetDiffFunc, GetSharedMask, GetLBILocation, \
    IsDiffFuncAvailable
from Bit2Edge.model.layer import CustomInput, CustomLayer, GetIdealNeurons, JoinLayer, ComputeSizeFromCustomLayer
from Bit2Edge.model.pool import ModelPool
from Bit2Edge.model.utils import (ConvertNumpyDenseToSparseTensor, EnforceToNumpyArray, SwitchDatatype, LoadModel,
                                  TryResetStateInTFModel, HasBatchNorm)
from Bit2Edge.utils.cleaning import RunGarbageCollection
from Bit2Edge.utils.file_io import FixPath
from Bit2Edge.utils.verify import InputFullCheck, TestState, TestStateByWarning, ValidateCondition

__FEATURE_v1__ = Union[ndarray]
__FEATURE_v2_1__ = Union[ndarray, sparse.SparseTensor]


class B2E_Model(DatasetLinker):
    """ 
    This class is a model designer which is built on top of TensorFlow 2.5+ with special modification.

    """

    def __init__(self, dataset: FeatureData, TF_Model: Optional[str], ModelKey: str):
        # [1]: Dataset Pointer
        super(B2E_Model, self).__init__(dataset=dataset)
        self.ModelKey: str = ModelKey
        self._TrainParams: TrainModelParams = TrainModelParams()
        self._state_: Dict[str, bool] = {'initialize': False, 'validate': False}

        # [2]: Function Selection
        self._function_: Dict[str, Union[Callable, List]] = self._GetFunctionTable_()[self.ModelKey]

        # [3]: Setup Model
        self._InterModel: Optional[Model] = None
        self._TF_Model: Optional[Model] = LoadModel(filepath=TF_Model, compilation=False)

        if isinstance(self._TF_Model, Model):
            self._ResetStateOnModel_()
            self._ValidateNewModel_()

    def BindNewDataset(self, dataset: FeatureData) -> None:
        super(B2E_Model, self).BindNewDataset(dataset=dataset)

    # [1]: Setup Self ----------------------------------------------------------------------------------------
    def _GetFunctionTable_(self) -> Dict[str, Dict]:
        # [1]: Dictionary containing much information
        # 0th: Build Function --- 1st: Get Input --- 2nd: The description name to retrieve last layer
        # BUT only the prefix (M, G, E) is more important and mostly applied.
        three = ['M-model', 'G-model', 'E-model']

        MODEL_DICT: Dict[str, Dict] = \
            {
                '02': {'build': self._Net02_, 'input': BE_2Input, 'output': three},  # 2 Inputs, 3 models
                '03': {'build': self._Net03_, 'input': BE_3Input, 'output': three},  # 3 Inputs, 3 models
            }

        return MODEL_DICT

    def _ValidateNewModel_(self):
        self._IsModelAvailable_(errno=True)

        warning('A new model may have been loaded successfully in class::B2E_Model.')
        OutputDtype = self._TF_Model.output.dtype
        if not isinstance(OutputDtype, np.dtype):
            OutputDtype = SwitchDatatype(OutputDtype)
        self._TrainParams.SetOutputDtype(value=OutputDtype)
        self._TrainParams.SetTargetSize(value=self._TF_Model.output[1])
        self._state_['validate']: bool = True

    # [2]: Model Function ----------------------------------------------------------------------------------------
    def _CalculateDensity_(self, inputs: List[ndarray]) -> None:
        print('-' * 26, 'CALCULATE DENSITY', '-' * 25)

        def MidAreaToLeftArea(value: float) -> float:
            return (100 - value) / 2

        def MidAreaToLeftMidArea(value: float) -> float:
            return value + MidAreaToLeftArea(value)

        # [1]: Compute Input Distributions
        distributions: List[Dict[str, float]] = []
        for i, array in enumerate(inputs):
            density: ndarray = np.count_nonzero(array, axis=1, keepdims=False)
            # Min-Max-Mean-Standard Deviation
            temp = {
                'shape': array.shape,
                'n': int(array.shape[0]),
                'size': int(array.shape[1]),
                'min': int(np.min(density)),
                'max': int(np.max(density)),
                'mean': float(np.mean(density)),
                'std': float(np.std(density)),
            }

            for value in ('90', '95', '97.5', '99', '99.5', '99.9', '99.99'):
                temp[f'{value}r'] = int(np.percentile(density, MidAreaToLeftMidArea(float(value)), axis=0))
                temp[f'{value}l'] = int(np.percentile(density, MidAreaToLeftArea(float(value)), axis=0))

            distributions.append(temp)
            print(f"Input #{i}: {temp}")

        # [2]: Compute whether we can convert data into sparse matrices
        IsInputSparse: List[bool] = [False] * len(inputs)
        if STRUCTURE.get('Attempt-Sparse', False):
            DENSITY_THRESHOLD: float = 7.5 * 1e-3
            for idx, array in enumerate(inputs):
                density: float = distributions[idx]['mean'] / inputs[idx].shape[1]
                # This threshold is opted because cuSparse can only out-perform with extremely sparse matrix
                # at threshold 99 % minimum (viewable performance may at 99.25-99.5 %) compared to cuBlas.
                # We have found two attempts that outperformed the cuSparse with smaller gap threshold.
                # >> Outperform at 98 % matrix sparsity and better cuSparse at most case (>= 99.5 % scenarios).
                # See papers:
                # 1) https://arxiv.org/pdf/2006.10901.pdf
                # 2) https://arxiv.org/pdf/2005.14469.pdf
                if density < DENSITY_THRESHOLD:
                    IsInputSparse[idx] = True

        self._TrainParams.SetInputDistribution(value=distributions)
        self._TrainParams.SetIsInputSparse(value=IsInputSparse)

    def _DisplayLevelOfDensity_(self) -> None:
        distributions = self._TrainParams.GetInputDistribution()
        print('Number of Inputs:', len(distributions))
        for idx, distribution in enumerate(distributions):
            minimum, maximum = distribution['min'], distribution['max']
            mean, std = distribution['mean'], distribution['std']
            inputSize: int = self._TrainParams.GetInputShape()[idx][1]
            print('-' * 25)
            print(f'Number of Non-Zero features over {inputSize} features: {minimum} (min) <--> {maximum} (max).')
            print(f'Distribution #1: {mean} ± {std}.')
            print(f'Distribution #2: {(mean / inputSize):.4f} (%) ± {(std / inputSize):.4f} (%).')
            for key, value in distribution.items():
                if key[-1] == 'r': # or key[-1] == 'l' (not both)
                    number = key[0:len(key) - 1]
                    left, right = distribution[number + 'l'], distribution[number + 'r']
                    print(f'Percentile {number} (%) left={left} and right={right}.')

        print('-' * 25)

    def GetInput(self, data: Tuple[__FEATURE_v1__, __FEATURE_v1__], force_dense: bool = False,
                 reverse: bool = False) -> List[__FEATURE_v2_1__]:
        getter: Callable = self._function_['input']
        StartLabelInfo, EndLabelInfo = self.Dataset().GetEnvLabelInfo(update=True)
        outcome = getter(*data, Start=StartLabelInfo, End=EndLabelInfo, reverse=reverse)

        if not STRUCTURE.get('Attempt-Sparse', False) or force_dense:
            return outcome

        SparseState = self._TrainParams.GetIsInputSparse()
        return [ConvertNumpyDenseToSparseTensor(out) if SparseState[idx] else out for idx, out in enumerate(outcome)]

    def Initialize(self, data: Tuple[__FEATURE_v1__, __FEATURE_v1__]) -> None:
        inputs: List = self.GetInput(data=data, force_dense=True)
        self._TrainParams.SetInputShape(value=[arr.shape for arr in inputs])

        self._CalculateDensity_(inputs=inputs)
        self._DisplayLevelOfDensity_()
        self._state_['initialize'] = True

    def compile(self, optimizer: Optimizer, loss: str, metrics: Optional[Union[str, List[str]]]) -> None:
        self._IsModelAvailable_(errno=True)
        self._ForceTraining_(errno=True)
        self._TF_Model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def build(self, TRAINING_DATA: Tuple[__FEATURE_v1__, __FEATURE_v1__], SampleSize: int,
              TargetSize: int = 1, model_dtype=DEFAULT_OUTPUT_NPDTYPE) -> Model:
        self._ForceTraining_()
        if not self._state_['initialize']:
            # This is to configure some data information such as input shape and input dtype.
            self.Initialize(data=TRAINING_DATA)

        if self._IsModelAvailable_():
            warning('Your model have already been initialized >> Disable execution.')
            return self._TF_Model

        self._TF_Model = self._function_['build'](SampleSize=SampleSize, TargetSize=TargetSize)
        self._TrainParams.SetOutputDtype(value=model_dtype)
        self._TrainParams.SetTargetSize(value=TargetSize)

        self._ValidateNewModel_()  # When model is set, validate to be True
        return self._TF_Model

    def fit(self, x_train: Tuple[__FEATURE_v1__, __FEATURE_v1__], y_train: __FEATURE_v1__,
            x_val: Optional[Tuple[__FEATURE_v1__, __FEATURE_v1__]] = None, y_val: Optional[__FEATURE_v1__] = None,
            callbacks: Optional[List] = None):
        # [1]: Initialize Configuration
        self._IsModelAvailable_(errno=True)
        self._ForceTraining_(errno=True)
        self._TF_Model.summary()
        TestState((x_val is None and y_val is None) or (x_val is not None and y_val is not None),
                  f'Validation set is not stable: X::{type(x_val)} and Y::{type(y_val)}')

        # [2]: Train your model
        RunGarbageCollection()

        train: Callable = self._TF_Model.fit
        SHUFFLE, VERBOSE, BATCH_SIZE = framework['Shuffle'], framework['verbose'], framework['Training Batch Size']
        START_EPOCH, NUM_EPOCH = framework['Initial Epoch'], framework['Maximum Epochs']
        WORKERS, QUEUE = framework['Workers'], framework['Queue']
        Frequency = framework['Frequency']

        if x_val is None and y_val is None:
            return train(x=self.GetInput(x_train), y=y_train, shuffle=SHUFFLE, callbacks=callbacks, verbose=VERBOSE,
                         batch_size=BATCH_SIZE, initial_epoch=START_EPOCH, epochs=NUM_EPOCH,
                         use_multiprocessing=False, workers=WORKERS, max_queue_size=QUEUE)

        if not HasBatchNorm(self.GetModel()):
            ValBatch = BATCH_SIZE
        else:
            ValBatch = int(BATCH_SIZE / y_train.shape[0] * y_val.shape[0])

        return train(x=self.GetInput(x_train), y=y_train, shuffle=SHUFFLE, callbacks=callbacks, verbose=VERBOSE,
                     batch_size=BATCH_SIZE, initial_epoch=START_EPOCH, epochs=NUM_EPOCH,
                     validation_freq=Frequency, validation_data=(self.GetInput(x_val), y_val),
                     validation_batch_size=ValBatch, use_multiprocessing=False, workers=framework['Workers'],
                     max_queue_size=framework['Queue'])

    def _CorePredict_(self, x, func: Callable, verbose: bool = True):
        if not self._state_['validate']:
            self._ValidateNewModel_()
        BATCH = framework[('Training Batch Size' if not HasBatchNorm(self.GetModel()) else 'Testing Batch Size')]
        RunGarbageCollection(0)
        return func(x=x, batch_size=BATCH, verbose=verbose or framework['verbose'],
                    max_queue_size=framework['Queue'], workers=framework['Workers'], use_multiprocessing=False)

    def _PredictInput_(self, func: Callable, data: Tuple[ndarray, ndarray], reverse: bool = False,
                       verbose: Optional[bool] = True):
        x = self.GetInput(data=data, force_dense=False, reverse=reverse)
        return self._CorePredict_(x, func=func, verbose=verbose)

    # --------------------------------------------------
    @staticmethod
    def _ModeStandard_(mode: int) -> int:
        InputFullCheck(mode, name='mode', dtype='int')
        TestState(mode in (0, 1, 2), 'The prediction mode must be either 0 or 1 or 2.')
        if mode != 0:
            TestStateByWarning(mode != 1, ' This model does not support (reverse) standardization.')
            mode: int = 0
        return mode

    def GetDataV2(self, data: Tuple[ndarray, ndarray], mode: int, toTensor: bool = False) \
            -> Tuple[Optional[List], Optional[List]]:
        mode = B2E_Model._ModeStandard_(mode=mode)
        RunGarbageCollection(1)
        FORWARD_DATA: Optional[List] = None
        REVERSE_DATA: Optional[List] = None

        def CastToTensor(x: ndarray):
            return convert_to_tensor(x, dtype=SwitchDatatype(x.dtype), dtype_hint=x.dtype)

        if mode in (0,):
            FORWARD_DATA = self.GetInput(data=data, force_dense=False, reverse=False)
            if toTensor:
                for idx, fdata in enumerate(FORWARD_DATA):
                    FORWARD_DATA[idx] = CastToTensor(fdata)
            return FORWARD_DATA, REVERSE_DATA

        if mode in (1,):
            REVERSE_DATA = self.GetInput(data=data, force_dense=False, reverse=True)
            if toTensor:
                for idx, rdata in enumerate(REVERSE_DATA):
                    REVERSE_DATA[idx] = CastToTensor(rdata)
            return FORWARD_DATA, REVERSE_DATA

        FORWARD_DATA = self.GetDataV2(data=data, mode=0, toTensor=toTensor)[0]

        FUNCTION_GETTER: Callable = self._function_['input']
        ReverseFunction = GetDiffFunc(func=FUNCTION_GETTER)

        if ReverseFunction is not None:  # Don't mapping if not toTensor
            StartLabelInfo, EndLabelInfo = self.Dataset().GetEnvLabelInfo(update=False)
            REVERSE_DATA: List = ReverseFunction(*data, Start=StartLabelInfo, End=EndLabelInfo,
                                                 original_reverse=False)
            for idx, value in enumerate(REVERSE_DATA):
                if value is not None:
                    REVERSE_DATA[idx] = CastToTensor(value)

            SharedMask = GetSharedMask(func=FUNCTION_GETTER)
            for index in SharedMask:
                REVERSE_DATA[index] = FORWARD_DATA[index]

        RunGarbageCollection(0)
        return FORWARD_DATA, REVERSE_DATA

    def _PredictV2_(self, fdata, rdata, func: Callable, mode: int = 0, verbose: Optional[bool] = True) -> ndarray:
        mode = B2E_Model._ModeStandard_(mode=mode)
        self._IsModelAvailable_(errno=True)

        if mode == 0:
            return EnforceToNumpyArray(self._CorePredict_(func=func, x=fdata, verbose=verbose))

        if mode == 1:
            return EnforceToNumpyArray(self._CorePredict_(func=func, x=rdata, verbose=verbose))

        y_pred_forward = self._CorePredict_(func=func, x=fdata, verbose=verbose)
        y_pred_backward = self._CorePredict_(func=func, x=rdata, verbose=verbose)
        return (EnforceToNumpyArray(y_pred_forward) + EnforceToNumpyArray(y_pred_backward)) / 2

    def PredictV2(self, fdata, rdata, mode: int = 2, verbose: Optional[bool] = True) -> ndarray:
        return self._PredictV2_(fdata=fdata, rdata=rdata, func=self._TF_Model.predict, mode=mode, verbose=verbose)

    def EvaluateV2(self, fdata, rdata, mode: int = 2, verbose: Optional[bool] = True) -> ndarray:
        return self._PredictV2_(fdata=fdata, rdata=rdata, func=self._TF_Model.evaluate, mode=mode, verbose=verbose)

    def _PredictV1_(self, data: Tuple[ndarray, ndarray], mode: int = 0, func: Optional[Callable] = None,
                    verbose: Optional[bool] = True) -> ndarray:
        """
        This method will predict the features in the argument :arg:`data` (if compatible).

        Arguments:
        ---------

        data : ndarray
            The data needed to be predicted.

        mode : int
            If :arg:`mode`=0, predict with that data. If :arg:`mode`=1 (if the model supported reverse
            standardization), predict feature in reverse mode. If :arg:`mode`=2, attempt to predict the
            result in standardization mode: If support standardization, the result is the average
            between :arg:`mode`=0 and :arg:`mode`=1.

        evaluateMode : bool
            If `func` is not provided, it used either the `evaluate` or `predict` of the _TF_Model.

        func : Callable
            If provided, this is the function applied to predict `data`.

        Returns:
        -------

        A numpy array

        """
        mode = B2E_Model._ModeStandard_(mode=mode)
        self._IsModelAvailable_(errno=True)

        if mode == 0:
            return EnforceToNumpyArray(self._PredictInput_(func=func, data=data, reverse=False, verbose=verbose))

        if mode == 1:
            return EnforceToNumpyArray(self._PredictInput_(func=func, data=data, reverse=True, verbose=verbose))

        y_pred_forward = self._PredictInput_(func=func, data=data, reverse=False, verbose=verbose)
        y_pred_backward = self._PredictInput_(func=func, data=data, reverse=True, verbose=verbose)
        return (EnforceToNumpyArray(y_pred_forward) + EnforceToNumpyArray(y_pred_backward)) / 2

    def PredictV1(self, data: Tuple[ndarray, ndarray], mode: int = 2, verbose: Optional[bool] = True) -> ndarray:
        return self._PredictV1_(data=data, mode=mode, func=self._TF_Model.predict, verbose=verbose)

    def EvaluateV1(self, data: Tuple[ndarray, ndarray], mode: int = 2, verbose: Optional[bool] = True) -> ndarray:
        return self._PredictV1_(data=data, mode=mode, func=self._TF_Model.evaluate, verbose=verbose)

    def GetInterModel(self) -> Model:
        self._IsModelAvailable_(errno=True)
        if self._InterModel is not None:
            return self._InterModel

        LAST_LAYERS: List = self.GetLastLayerName()
        TestState(isinstance(LAST_LAYERS, list), 'There are no final combination layers found >> Check your model.')
        SUB_LAYERS = Concatenate(name='inter_concatenate')([self._TF_Model.get_layer(layer).output
                                                            for layer in LAST_LAYERS])
        self._InterModel = Model(inputs=self._TF_Model.input, outputs=SUB_LAYERS)
        return self._InterModel

    def GetLastLayerName(self, verbose: bool = False) -> Optional[List[str]]:
        # See B2E_Model._SubModelToFinal_()
        PrefixLayerNames: List[str] = self.GetPrefixLabelNameOfSubModel()
        # SerializedLayerNames: List[str] = self.GetSerializedLabelNameOfSubModel()
        LAST_LAYERS = []
        for idx, layer in enumerate(self._TF_Model.layers):
            if not isinstance(layer, Dense):
                continue
            if verbose:
                MSG: str = f'Layer Idx #{idx}'
                for w in layer.get_weights():
                    MSG += f' - Shape: {w.shape}'
                print(MSG)
            if layer.get_weights()[1].shape[-1] == 1 and '_' in layer.name:
                # See function B2E_Model._OptName_() and B2E_Model._SubModelToFinal_() for more details
                CurrentLayerNamePrefix = str(layer.name).split('_')[0]
                if any(prefix in CurrentLayerNamePrefix for prefix in PrefixLayerNames):
                    LAST_LAYERS.append(layer.name)
        return LAST_LAYERS if len(LAST_LAYERS) != 0 else None

    def GetLastLayerOutput(self, data: Tuple[ndarray, ndarray], mode: int = 2,
                           verbose: Optional[bool] = True) -> ndarray:
        InterModel = self.GetInterModel()
        return self._PredictV1_(data=data, mode=mode, func=InterModel.predict, verbose=verbose)

    def GetLabelNamesOfSubModel(self) -> List[str]:
        return self._function_['output']

    def GetPrefixLabelNameOfSubModel(self) -> List[str]:
        return [name.split('-')[0] for name in self.GetLabelNamesOfSubModel()]

    def GetSerializedLabelNameOfSubModel(self) -> List[str]:
        return [name.replace('-', '_') for name in self.GetLabelNamesOfSubModel()]

    # [3]: Model Support ----------------------------------------------------------------------------------------
    def _IsModelAvailable_(self, errno: bool = False) -> bool:
        result: bool = isinstance(self._TF_Model, Model)
        return ValidateCondition(result, errno=errno, msg='The model is not initialized yet.')

    def _ForceTraining_(self, errno: bool = False) -> bool:
        condition: bool = True
        if self._TF_Model is not None:
            condition: bool = self._IsModelAvailable_(errno=False) and self._TF_Model.trainable
        return ValidateCondition(self.Dataset().trainable and condition, errno=errno,
                                 msg='The model is not in training phase -> Stop here.')

    def _ResetStateOnModel_(self) -> None:
        if self._TF_Model is None:
            return None
        if self.Dataset().retrainable or not self.Dataset().trainable:
            TryResetStateInTFModel(self._TF_Model)
        self._TF_Model.trainable = self.Dataset().trainable

    def GetModel(self) -> Optional[Model]:
        self._IsModelAvailable_(errno=False)
        return self._TF_Model

    def SetModel(self, filepath: str, training: Optional[bool]) -> bool:
        if not ModelPool.IsValidFilePath(filepath=filepath):
            return False

        Saved_TF_Model = self._TF_Model
        if self._TF_Model is not None:
            self._TF_Model = None

        trainable: bool = self.Dataset().trainable
        if training is not None:
            trainable = training
        try:
            self._TF_Model = LoadModel(filepath=FixPath(FileName=filepath, extension='.h5'), compilation=trainable)
            self._state_['validate'] = False
            self._ResetStateOnModel_()
            self._ValidateNewModel_()
            state: bool = True
        except OSError:
            self._TF_Model = Saved_TF_Model
            state: bool = False

        if self._InterModel is not None:
            self._InterModel = None
            self.GetInterModel()
        RunGarbageCollection(0)  # TensorFlow cannot free memory automatically
        return state

    def SetWeights(self, new_weights: Union[str, Model], by_name: bool = False,
                   skip_mismatch: bool = False) -> bool:
        """
        Loads all layer weights, either from a TensorFlow or an HDF5 weight file.
        References: https://www.tensorflow.org/api_docs/python/tf/keras/Model#load_weights

        Arguments:
        ---------
        
        new_weights : str, Model
            path to the weights file to load.

        by_name : bool
            If False, weights are loaded based on the network's topology or the architecture should be
            the same as when the weights were saved. For layers with no weights are not taken into
            account in the topological ordering, so adding or removing these layers is fine. If True,
            weights are loaded into layers only if they share the same name. This is useful for fine-tuning
            or transfer-learning models where some layers have changed. Only topological loading (arg::by_name
            =False) is supported when loading weights from the TensorFlow format. Note that topological loading
            differs slightly between TensorFlow and HDF5 formats. Default to False.

        skip_mismatch: bool
            Whether to skip loading of layers where there is a mismatch in the number of weights,
            or a mismatch in the shape of the weight (only valid when by_name=True). Default to False.

        Returns:
        -------

        status : bool
            Showing the state whether the assignment worked properly without any error.

        """
        # [0]: Parameters Validation
        InputFullCheck(by_name, name='by_name', dtype='bool', delimiter='-')
        InputFullCheck(skip_mismatch, name='skip_mismatch', dtype='bool', delimiter='-')
        TestState(self._IsModelAvailable_(), 'No _TF_Model has been set yet.')
        TestStateByWarning(self._InterModel is None, 'The Inter_Model is found. This would be removed.')

        SUCCESS: bool = True
        if self._InterModel is not None:
            del self._InterModel
            self._InterModel = None
        RunGarbageCollection(0)

        try:
            if isinstance(new_weights, str):
                path: str = FixPath(FileName=new_weights, extension='.h5')
                self._TF_Model.load_weights(path, by_name=by_name, skip_mismatch=skip_mismatch)
            else:
                self._TF_Model.set_weights(weights=new_weights.get_weights())
            self._ValidateNewModel_()
        except OSError:
            SUCCESS = False
        return SUCCESS

    # ---------------------------------------------------- BUILD --------------------------------------------
    @staticmethod
    def _OptName_(basename: str, TargetSize: int = 1, index: Optional[int] = None) -> str:
        NAME = basename.replace('-', '_')
        NOTION = STRUCTURE.get('Predict-Notion', None)
        TestState(NOTION is not None, 'The prediction notion is not found, please check the config file.')
        TestState(len(NOTION) == len(set(NOTION)),
                  msg='The prediction notion is not unique, please check the config file.')
        if TargetSize == 1:
            return f"{NAME}_{NOTION[STRUCTURE.get('Predict-Notion-Index', 0)]}" if NOTION else NAME

        return f"{NAME}_{NOTION[index]}" if NOTION else f"{NAME}_{index + 1}"

    def ComputeModelOutputSize(self) -> int:
        self._IsModelAvailable_(errno=True)
        TempModel: Model = self._TF_Model.GetModel()
        OUTPUT_LAYER = TempModel.layers[-1].output
        TestState(isinstance(OUTPUT_LAYER, (Concatenate, Dense)),
                  'The last layer of the model must be either Concatenate or Dense layer.')

        if isinstance(OUTPUT_LAYER, Concatenate):
            TestState(OUTPUT_LAYER.name == 'output', msg='The provided model did not follow the correct standard.')

        return sum(1 for layer in TempModel.layers if layer.name.startswith('output_'))

    # ---------------------------------------------------
    # Build Section <- Utility Only
    def _GetInputLayers_(self, names: Union[List[str], Tuple[str, ...]]) -> List[Layer]:
        InputShape = self._TrainParams.GetInputShape()
        TestState(len(names) == len(InputShape), f'The name should be match with (={len(InputShape)}).')
        SparseState = self._TrainParams.GetIsInputSparse()
        if SparseState is None:
            SparseState = [False] * len(InputShape)
        return [CustomInput(size=InputShape[i][1], name=name, sparseState=SparseState[i])
                for i, name in enumerate(names)]

    def _SubModelToFinal_(self, FinalLayerSetup: Dict[str, Tuple[Any, str, str]], TargetSize: int):
        """
            This method is to join all sub-models on each prediction target into one final model.
            If there are one or more prediction, the final model will be the Concatenate layer.
            If there is only one prediction, the final model will be the Dense layer only.

            Assuming the userConfig.MODEL_STRUCTURE is correctly loaded (which should be), and the 
            selected target is ordered and prioritized by BDE-BDFE-..., with three sub-models 
            (by now a default) then:

            For the base network, the layer can derive the vector representation is G_Vect,
            M_Vect, E_Vect.

            If there are one prediction only, the layers' name are:
            - Sub-model (Dense): G_BDE, M_BDE, E_BDE
            - 1 Concatenate
            - Prediction: 'output_BDE' or 'output_1'
            
            If there are two or more predictions, the layers' names are:
            - Sub-model (Dense): (G_BDE, M_BDE, E_BDE), (G_BDFE, M_BDFE, E_BDFE), ...
            - 'n' Concatenate (each) 
            - Prediction: 'output_BDE' or 'output_1', 'output_BDFE' or 'output_2', ...
            - Join: 'output'

        """
        TestState(TargetSize != 0, 'The number of target for prediction cannot be zero.')
        JOIN_LAYERS = []
        for layer_name in self.GetLabelNamesOfSubModel():
            TestState(layer_name in FinalLayerSetup, f'LAYER={layer_name} is not found, raising incompatible.')

        NOTION = STRUCTURE.get('Predict-Notion', None)
        TestState(NOTION is not None, 'The prediction notion is not found, please check the config file.')
        TestState(len(NOTION) == len(set(NOTION)),
                  msg='The prediction notion is not unique, please check the config file.')
        for idx in range(TargetSize):
            LAYERS = []
            for name in self.GetLabelNamesOfSubModel():
                layer, activation, NAME = FinalLayerSetup[name]
                if name.find(NAME, 0, len(NAME)) == -1:
                    warning('The layer prefix is not correct, please check the layer-setup variable.')
                    # String checking, if the name's is not correct, 
                    # re-using the layer's default name as the default prefix.
                    NAME = name
                LAYERS.append(CustomLayer(layer, units=1, activation=STRUCTURE[activation],
                                          name=B2E_Model._OptName_(NAME, TargetSize, index=idx)))
            if TargetSize == 1:
                FinalName = f"output_{NOTION[STRUCTURE.get('Predict-Notion-Index', 0)]}" \
                    if NOTION else 'output_1'
                return JoinLayer(layers=LAYERS, TargetSize=1, name=FinalName)
            else:
                FinalName = f"output_{NOTION[idx]}" if NOTION else f"output_{idx + 1}"
                layer = JoinLayer(layers=LAYERS, TargetSize=1, name=FinalName)
                JOIN_LAYERS.append(layer)

        return Concatenate(name='output')(JOIN_LAYERS)

    def _GModel_(self, INPUTS: List, TargetSize: int) -> Layer:
        LBI: int = GetLBILocation(func=self._function_['input'])
        GAct = STRUCTURE['G-model Core-Act']
        LBI_Size: int = self._TrainParams.GetInputShape()[LBI][1]

        GBaseLayer = CustomLayer(INPUTS[LBI], LBI_Size, activation=GAct, TargetOffset=TargetSize)
        GLayer = CustomLayer(GBaseLayer, LBI_Size, activation=GAct, TargetOffset=TargetSize)
        GLayer = CustomLayer(GLayer, LBI_Size, activation=GAct, TargetOffset=TargetSize)
        GLayer = Concatenate(name='G_Vect')([GBaseLayer, GLayer])

        return GLayer

    def _MModel_(self, INPUTS: List, IndexPaths: Tuple[int, ...], TargetSize: int) -> Tuple[Layer, Layer]:
        """ This is to output the first and final Dense layer of the M-model. """
        MAct = STRUCTURE['M-model Core-Act']
        SampleSize = self._TrainParams.GetInputShape()[0][0]
        FeatureSize = sum(self._TrainParams.GetInputShape()[idx][1] for idx in IndexPaths)
        print(f'Matrix Size for M-model: ({SampleSize}, {FeatureSize} -> {TargetSize}).', end='')
        SM_Neurons = GetIdealNeurons(FeatureDistributions=self._TrainParams.GetInputDistribution(), Indices=IndexPaths,
                                     TargetSize=TargetSize)

        MDrop = STRUCTURE['M-model Dropout']
        NAME: str = 'M_Vect'
        if len(IndexPaths) == 1:
            M_BDE = INPUTS[IndexPaths[0]]
        else:
            M_BDE = Concatenate(name='cluster')([INPUTS[idx] for idx in IndexPaths])
        M_BDE = CustomLayer(M_BDE, SM_Neurons[0], name=(NAME if len(SM_Neurons) == 1 else None),
                            activation=MAct, dropout=MDrop[0])
        if len(SM_Neurons) == 1:
            return M_BDE, M_BDE

        M1_BDE = CustomLayer(M_BDE, SM_Neurons[1], name=(NAME if len(SM_Neurons) == 2 else None),
                             activation=MAct, dropout=MDrop[1])
        if len(SM_Neurons) > 2:
            for idx, node in enumerate(SM_Neurons[2:], start=2):
                M1_BDE = CustomLayer(M1_BDE, node, name=(NAME if idx == len(SM_Neurons) - 1 else None),
                                     activation=MAct, dropout=MDrop[idx])
        return M_BDE, M1_BDE

    def _EModelStruct_(self, INPUTS: List, TargetSize: int, ENode: List[int]) -> List:
        LBI: int = GetLBILocation(func=self._function_['input'])
        EAct: str = STRUCTURE['E-model Core-Branch-Act']
        COEF = (STRUCTURE['E-model Bond-Env Scaling'], STRUCTURE['E-model LBondInfo Scaling'])
        LAYERS = []
        for idx, input_layer in enumerate(INPUTS):
            UNITS: int = ENode[idx] * COEF[(0 if idx != LBI else 1)]
            layer = CustomLayer(input_layer, units=UNITS, activation=EAct, TargetOffset=TargetSize)
            size: int = ComputeSizeFromCustomLayer(units=UNITS, TargetOffset=TargetSize)
            LAYERS.append({'layer': layer, 'index': idx, 'width': size})
        return LAYERS

    def _EModelJoinP1_(self, BondEnvLayer: Dict, LBILayer: Dict, TargetSize: int, ENode: List[int],) -> Dict:
        # [1]: Prepare calculation arguments
        EAct: str = STRUCTURE['E-model Core-Branch-Act']
        COEF = (STRUCTURE['E-model Bond-Env Scaling'], STRUCTURE['E-model LBondInfo Scaling'])
        LBI_Node: int = ENode[LBILayer['index']] * COEF[1]

        # [2]: Compute number of neurons for joining
        BE_Node = ENode[BondEnvLayer['index']] * COEF[0]
        distributions: List[Dict] = self._TrainParams.GetInputDistribution()
        NOM = distributions[BondEnvLayer['index']]['99r'] * BE_Node + \
              distributions[LBILayer['index']]['99r'] * LBI_Node
        DENOM = BE_Node + LBI_Node
        JOINING_NODE: int = int(NOM // (DENOM * 2))

        E_BDE = Concatenate()([BondEnvLayer['layer'], LBILayer['layer']])
        E_BDE = CustomLayer(E_BDE, JOINING_NODE, activation=EAct, TargetOffset=TargetSize)

        return {'layer': E_BDE, 'width': JOINING_NODE}

    def _EModelMerge_(self, ELayers: List[Dict], TargetSize: int, ENode: List[int],
                      EmbedLayer: Optional[Tuple[Layer, int]]) -> Layer:
        # [1]: Prepare calculation arguments
        LBI: int = GetLBILocation(func=self._function_['input'])
        LBI_Node: int = ENode[LBI] * STRUCTURE['E-model LBondInfo Scaling']
        EAct: str = STRUCTURE['E-model Core-Branch-Act']

        # [2]: Join the layer
        temp = [value['layer'] for value in ELayers]
        sizing: int = len(temp) + 1     # Include LBI path.
        if EmbedLayer is not None:
            temp.append(EmbedLayer[0])
            sizing += len(GetSharedMask(func=self._function_['input'])) - 1 + \
                      int(IsDiffFuncAvailable(func=self._function_['input']))

        # New Sizing
        # sizing = len(temp) + 1        # Include LBI path.
        # if EmbedLayer is not None:
        # sizing += EmbedLayer[1] - 1   # Remove duplicate LBI path.

        E_BDE = Concatenate()(temp)
        E_BDE = CustomLayer(E_BDE, LBI_Node * sizing, activation=EAct, TargetOffset=TargetSize)
        return CustomLayer(E_BDE, units=(LBI_Node * sizing) // 2, name='E_Vect',
                           activation=STRUCTURE['E-model Merge-Act'], TargetOffset=TargetSize)

    # --------------------------------------------------------------------------------------------------------
    # Build Section <- Completed Model
    def _BaseNet01_(self, INPUTS: List[Layer], SampleSize: int, TargetSize: int):
        """
            This function describe simply how we construct our Bit2Edge's model using the same template.
            It's strongly binding with all input getter function in file data.py so small modification
            can impact the model's meaning. See model's position in data.py
        """
        LBI: int = GetLBILocation(func=self._function_['input'])  # Constant requirement

        # [1.2]: Calculate Neurons for Error Model
        density_level = STRUCTURE['E-model Type']
        ENode = [int(distribution[density_level])
                 for idx, distribution in enumerate(self._TrainParams.GetInputDistribution())]
        print(f'---> Density-based Error Node: {ENode}.')

        # [2.1]: G-Model
        G_BDE = self._GModel_(INPUTS=INPUTS, TargetSize=TargetSize)

        # [2.2]: M-Model
        IndexPaths: Tuple[int, ...] = (0, 1, 2)
        M_FirstLayer, M_BDE = self._MModel_(INPUTS=INPUTS, IndexPaths=IndexPaths, TargetSize=TargetSize)

        # [2.3]: E-Model (Init << Top)
        EBaseLayers = self._EModelStruct_(INPUTS=INPUTS, TargetSize=TargetSize, ENode=ENode)

        # [2.3.1]: E-Model (Aggregation << Medium)
        LAYERS = []
        for idx, layer in enumerate(EBaseLayers):
            if idx == LBI:
                continue
            temp = self._EModelJoinP1_(BondEnvLayer=layer, LBILayer=EBaseLayers[LBI],
                                       TargetSize=TargetSize, ENode=ENode)
            LAYERS.append(temp)

        EmbedLayer = M_FirstLayer if STRUCTURE.get('E-model Mapping First M-model Layer', None) else None

        # [2.3.2]: E-Model (Merge << Bottom)
        E_BDE = self._EModelMerge_(ELayers=LAYERS, TargetSize=TargetSize, ENode=ENode,
                                   EmbedLayer=(EmbedLayer, len(IndexPaths)))

        # [3]: Full Model
        # First value is the layer, second value is the selected activation function
        # Third-value is the prefix of the layer's name
        SETUP: Dict[str, Tuple[Layer, str, str]] = \
            {
                'G-model': (G_BDE, 'G-model Last-Act', 'G'),
                'E-model': (E_BDE, 'E-model Last-Act', 'E'),
                'M-model': (M_BDE, 'M-model Last-Act', 'M'),
            }

        OUTPUT = self._SubModelToFinal_(FinalLayerSetup=SETUP, TargetSize=TargetSize)
        return Model(inputs=INPUTS, outputs=OUTPUT)

    def _Net02_(self, SampleSize: int, TargetSize: int) -> Model:
        INPUTS = self._GetInputLayers_(names=('BEnv_4', 'BEnv_2', 'LBondInfo'))
        return self._BaseNet01_(INPUTS=INPUTS, SampleSize=SampleSize, TargetSize=TargetSize)

    def _Net03_(self, SampleSize: int, TargetSize: int) -> Model:
        INPUTS = self._GetInputLayers_(names=('BEnv_4', 'BEnv_2', 'LBondInfo', 'BEnv_6'))
        return self._BaseNet01_(INPUTS=INPUTS, SampleSize=SampleSize, TargetSize=TargetSize)
