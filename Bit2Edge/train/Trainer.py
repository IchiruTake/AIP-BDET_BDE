# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from logging import warning, info
from time import perf_counter, sleep
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

# import keras
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from numpy import ndarray

from Bit2Edge.config.devConfig import LR_Wrapper, GetLearningRate
from Bit2Edge.config.modelConfig import FRAMEWORK, TRAIN_CALLBACKS, MODEL_STRUCTURE
from Bit2Edge.dataObject.DataBlock import GetDtypeOfData
from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.dataObject.DatasetLinker import DatasetLinker
from Bit2Edge.dataObject.InputState import InputState
from Bit2Edge.model.model import B2E_Model
from Bit2Edge.train.TrainerUtils import DrawNeuralProgress, GetOptimizer, GetCallbacks, SaveModel, GetLossAndMetric, \
    _ShowPlot
from Bit2Edge.utils.cleaning import RunGarbageCollection
from Bit2Edge.utils.file_io import FixPath, ExportFile
from Bit2Edge.utils.verify import MeasureExecutionTime, InputFullCheck, InputCheckRange, TestState

STR = Optional[str]


class Trainer(DatasetLinker):
    """
        This class is a controlled class in which you use it to train the model only.
        After initialization, you run method `prepare()` to split out your data. And
        call `TrainAPI()` to finish the task.
    """

    @MeasureExecutionTime
    def __init__(self, dataset: FeatureData, StorageFolder: str, FP16: bool = False, JIT: bool = False):
        """
        Arguments:
        ---------

        dataset : FeatureData
            The FeatureData's object where we stored our feature

        StorageFolder : str
            The directory in which we stored our result.

        FP16 : bool
            If True, the datatype when train the model is 'float16'. Default to False ('float32').

        JIT : bool
            If True, the optimizer will enable JIT train (useful for ConvNet-style or RNN/LSTM-Transformer-style).
            Default to False.

        """
        super(Trainer, self).__init__(dataset=dataset)

        # [1]: Dataset Pointer
        print('-' * 27, 'Configure Trainer', '-' * 27)
        TestState(self.Dataset().trainable and self.GetKey() == 'Train',
                  'The :arg:`dataset` is not trainable or in-valid.')

        # [2]: Attribute for Model
        self._ModelState: Dict[str, Union[int, bool, Any]] = \
            {
                'dynamic': False, 'JIT': JIT,
                'dtype': f'float{(16 if FP16 else 32)}',
            }
        if FP16:
            try:
                try:
                    from tensorflow.keras import mixed_precision
                except (ImportError, ImportWarning):
                    from tensorflow.python.keras import mixed_precision
                try:
                    policy = mixed_precision.experimental.Policy('mixed_float16')
                    mixed_precision.experimental.set_policy(policy)
                except (RuntimeError, ValueError, ImportError, ImportWarning):
                    policy = mixed_precision.Policy('mixed_float16')
                    mixed_precision.set_policy(policy)

            except (RuntimeError, ValueError, ImportError, ImportWarning):
                warning('Mixed-Precision is not enabled in this GPU.')

        # [3]: Export Data
        StringQuickCheck: Callable = lambda string: string is not None and isinstance(string, str) and string != ""
        self._StorageFolder: str = FixPath(StorageFolder, extension='/') if StringQuickCheck(StorageFolder) else ""
        self.TF_Model: Optional[B2E_Model] = None

    # ----------------------------------------------------------------------------------------------------------------
    def InitializeModel(self, ModelKey: str, TF_Model: STR = None) -> None:
        """
        This method is to initialize the TensorFlow's Model.
        Arguments:
        ---------

        ModelKey : int
            The configuration of the model (declared in model.py), which determined the architecture of the model

        TF_Model : str
            If set, this is the directory of the trained model where we want to retrain
            our pre-trained model. Default to None
        """
        if self.Dataset().retrainable:
            info('The given dataset required fine-tuning (re-train model). Please be careful.')
            TestState(TF_Model is not None, 'If retrained the model, the model must be provided beforehand.')
        self.TF_Model = B2E_Model(dataset=self.Dataset(), TF_Model=TF_Model, ModelKey=ModelKey)

    def TrainAPI(self, SavedModelName: Tuple[STR, STR] = ('Initial Model', 'Final-Epoch Model'),
                 useBestValidation: bool = True, parallel: bool = False, SaveOpt: bool = False,
                 SkipSketchPoint: int = 2) -> None:
        """
        This method is the API to train a neural network model

        Arguments:
        ---------

        SavedModelName : Tuple[str, str]
            The name of your model would be saved. The first argument is used at the first epoch, while the
            second argument is used at the last epoch. All models are stored by the :arg:`StorageFolder`.
            Default to ('Initial Model', 'Final-Epoch Model')

        useBestValidation : bool
            If True and the model used the Early Stopping, it selected the best model that has been performed
            on validation data (data can be either trained or not) to apply on the test set.

        parallel : bool
            If True, it created the :meth:`tf.distribute.MirroredStrategy()` to train your model in parallel.
            Default to False.

        SaveOpt : bool
            If True, the model would save the optimizer on both the initial-epoch model and the final-epoch
            model. Default to False

        SkipSketchPoint : int
            The number of epochs at the beginning which is skipped to visualize the loss. Default to 2
            if it is the new model; otherwise zero.

        """
        # Hyper-parameter Verification
        if True:
            InputFullCheck(useBestValidation, name='useBestValidation', dtype='bool')
            InputCheckRange(SkipSketchPoint, name='SkipSketchPoint', maxValue=FRAMEWORK['Maximum Epochs'])
            if self.Dataset().retrainable:
                SkipSketchPoint = 0

            InputFullCheck(value=parallel, name='parallel', dtype='bool')
            InputFullCheck(value=SaveOpt, name='SaveOpt', dtype='bool')
            TestState(isinstance(self.TF_Model, B2E_Model), 'The _TF_Model is NOT initialized yet.')

        print('-' * 80)
        print('Bit2Edge is train your model.')
        # tf.debugging.set_log_device_placement(True)
        start: float = perf_counter()

        # [0]: Prepare Dataset
        TRAIN, VAL = self._Prepare_()
        if self.Dataset().GetState().get('Duplicate', None) is not None:
            SampleSize: int = self.Dataset().GetState().get('Duplicate')['Train'][0]
        else:
            SampleSize: int = TRAIN[0].shape[0]

        # [1]: Training up Model
        print(f'Tensorflow/TF Version: {tf.__version__}.')
        print(f'Tensorflow: Git Version: {tf.__git_version__} --- Python Compiler Version: {tf.__compiler_version__}')

        if parallel:  # Should not be called
            strategy = tf.distribute.MirroredStrategy()  # Create a MirroredStrategy.
            print(f'Number of devices: {strategy.num_replicas_in_sync}.')
            with strategy.scope():
                self._InitModel_(SampleSize=SampleSize, TRAIN=TRAIN, InitModelName=SavedModelName[0], SaveOpt=SaveOpt)
                train_logs = self._Train_(TRAIN=TRAIN, VAL=VAL, FinalModelName=SavedModelName[1], SaveOpt=SaveOpt)
        else:
            self._InitModel_(SampleSize=SampleSize, TRAIN=TRAIN, InitModelName=SavedModelName[0], SaveOpt=SaveOpt)
            train_logs = self._Train_(TRAIN=TRAIN, VAL=VAL, FinalModelName=SavedModelName[1], SaveOpt=SaveOpt)

        # [2]: Finalize and Save the model
        self._Record_(training_logs=train_logs, SketchingPoint=SkipSketchPoint)

        CACHE = {}
        final_pred = self._Evaluation_(FinalModel=SavedModelName[1], data_cache=CACHE)
        pred = self._Predict_(BestValidationModel=useBestValidation, data_cache=CACHE, train_logs=train_logs,
                              useCkpt=True)
        pred_v2 = self._Predict_(BestValidationModel=useBestValidation, data_cache=CACHE, train_logs=train_logs,
                                 useCkpt=False)

        self._Sketch_(title='Model Performance', y_pred=pred if pred is not None else final_pred)
        print(f'Bit2Edge: Execution Time: {perf_counter() - start:.4f} (s).')

    def _Prepare_(self) -> Tuple:
        print('---> [1]: Retrieve dataset.')
        TRAIN = (self.Dataset().GetDataBlock('EnvData').GetData('Train'),
                 self.Dataset().GetDataBlock('LBIData').GetData('Train'),
                 self.Dataset().GetDataBlock('Target').GetData('Train'))

        print(f'Number of Observations: {TRAIN[0].shape[0]}.')
        VAL = (self.Dataset().GetDataBlock('EnvData').GetData('Val'),
               self.Dataset().GetDataBlock('LBIData').GetData('Val'),
               self.Dataset().GetDataBlock('Target').GetData('Val'))
        FoundNone: bool = any(feature is None for feature in VAL)
        if FoundNone:
            VAL = (None, None, None)
        return TRAIN, VAL

    def _InitModel_(self, SampleSize: int, TRAIN: Tuple[ndarray, ...], InitModelName: str, SaveOpt: bool):
        print('---> [2]: Initialize the Bit2Edge model.')
        lr: Optional[float] = None
        if FRAMEWORK['Initial Epoch'] != 0:
            # Update Learning Rate according to Learning Rate Scheduler
            for epoch in range(0, FRAMEWORK['Initial Epoch']):
                lr = LR_Wrapper(epoch=epoch, learning_rate=lr, verbose=False)
            lr = GetLearningRate(epoch=-1)

        if self._ModelState['JIT'] is True:
            tf.config.optimizer.set_jit(self._ModelState['JIT'])

        print('---> [2.5]: Constructing Model.')
        TrainEnv, TrainLBI, TrainTarget = TRAIN
        self.TF_Model.build(TRAINING_DATA=(TrainEnv, TrainLBI), SampleSize=SampleSize, TargetSize=TrainTarget.shape[1],
                            model_dtype=self._ModelState['dtype'])
        model = self.TF_Model.GetModel()
        model.trainable = True
        loss, metric = GetLossAndMetric(loss=FRAMEWORK['Loss Function'])
        opt = GetOptimizer(learning_rate=lr)
        model.compile(optimizer=opt, loss=loss, metrics=list(metric))

        if not self.Dataset().retrainable:
            if isinstance(InitModelName, str) and FRAMEWORK['Initial Epoch'] == 0:
                FileName: str = f'{self._StorageFolder}{InitModelName}'
                FileWeightName: str = f'{self._StorageFolder}[Weight] {InitModelName}'
                SaveModel(model, name=(FileName, FileWeightName), SaveOpt=SaveOpt, save_traces=True)
        else:
            warning('We attempted to reload the weights for retraining.')
            model.load_weights(filepath=model)
        return None

    def _Train_(self, TRAIN: Tuple[ndarray, ...], VAL: Tuple[ndarray, ...],
                FinalModelName: str, SaveOpt: bool):
        model = self.TF_Model.GetModel()

        print('---> [3]: Bit2Edge: Begin Learning. Please hold for a secs ... ')
        RunGarbageCollection(0)  # Memory must be free first before train
        callbacks = GetCallbacks(storage=self._StorageFolder, validation_checkpoint=all(v is not None for v in VAL))
        TrainEnv, TrainLBI, TrainTarget = TRAIN
        ValEnv, ValLBI, ValTarget = VAL
        # See Trainer._Prepare_()
        x_val = (ValEnv, ValLBI) if ValEnv is not None and ValLBI is not None else None
        t1 = perf_counter()
        hist = self.TF_Model.fit(x_train=(TrainEnv, TrainLBI), y_train=TrainTarget, x_val=x_val, y_val=ValTarget,
                                 callbacks=callbacks)
        RunGarbageCollection(0)
        print(f'Overall Training Time: {perf_counter() - t1:.4f} (s).')
        print('---> [4]: Bit2Edge: Finished Learning --> Saving the models & weights.')
        model.trainable = False
        if isinstance(FinalModelName, str):
            FileName: str = f'{self._StorageFolder}{FinalModelName}'
            FileWeightName: str = f'{self._StorageFolder}[Weight] {FinalModelName}'
            SaveModel(model, name=(FileName, FileWeightName), SaveOpt=SaveOpt, save_traces=True)
        return hist.history

    def _Record_(self, training_logs: Dict[str, List[float]], SketchingPoint: int) -> None:
        print('---> [5]: Bit2Edge: Record model performance.')
        DrawNeuralProgress(training_logs=training_logs, starting=SketchingPoint)
        start: int = FRAMEWORK['Initial Epoch']
        index: list = list(range(start, len(training_logs['loss']) + start))
        ExportFile(DataFrame=pd.DataFrame(data=training_logs, index=index),
                   FilePath=f'{self._StorageFolder}Histogram Progress.csv', index=True, index_label='Epoch #')

    def __OptModel__(self, CKPT_NAME: str, is_weight_file: bool, training: bool = False,
                     by_name: bool = False, skip_mismatch: bool = False) -> bool:
        if not is_weight_file:
            return self.TF_Model.SetModel(filepath=f'{self._StorageFolder}{CKPT_NAME}', training=training)

        return self.TF_Model.SetWeights(new_weights=f'{self._StorageFolder}{CKPT_NAME}',
                                        by_name=by_name, skip_mismatch=skip_mismatch)

    def __OpBestModel__(self, weight_mode: bool, train_logs: Dict[str, List[float]], useCkpt: bool = False) -> bool:
        print('-' * 25, f'Weight-Mode: {weight_mode}', '-' * 25)
        if 'val_loss' in train_logs:
            log_key = ('val_loss', min)
        elif 'val_acc' in train_logs:
            log_key = ('val_acc', max)
        elif 'loss' in train_logs:
            log_key = ('loss', min)
        else:
            raise ValueError
        TEMP: List[float] = train_logs[log_key[0]]
        BEST_EPOCH: int = TEMP.index(log_key[1](TEMP)) + 1  # 0-based index
        msg = f'The best model is found at epoch {BEST_EPOCH}'
        if len(TEMP) >= FRAMEWORK['Unchanged Epochs']:
            BEST_EPOCH_BY_CKPT: int = len(TEMP) - FRAMEWORK['Unchanged Epochs']
        else:
            BEST_EPOCH_BY_CKPT: int = len(TEMP)
        if BEST_EPOCH != BEST_EPOCH_BY_CKPT:
            msg += f' but the "best" by checkpoint is at ({BEST_EPOCH_BY_CKPT}).'
        print(msg)
        for idx, (ckptName, ckpt) in enumerate(TRAIN_CALLBACKS['Checkpoint'].items()):
            try:
                GetWeights: bool = ckpt.get('save_weights_only', False)
            except ValueError:
                continue
            if GetWeights != weight_mode:
                continue
            if 'epoch' in ckptName:
                # Switch to BEST_EPOCH_BY_CKPT rather than BEST_EPOCH
                ckptName = ckptName.format(epoch=BEST_EPOCH_BY_CKPT if useCkpt else BEST_EPOCH)

            print(f'Epoch: {BEST_EPOCH} -- Checkpoint Model Filename #{idx + 1}: {ckptName} .')
            if self.__OptModel__(CKPT_NAME=ckptName, is_weight_file=weight_mode,
                                 training=False, by_name=False, skip_mismatch=False):
                return True

        warning('Model cannot be found under class::Trainer. EXIT from here.')
        return False

    def _Predict_(self, BestValidationModel: bool, data_cache: Dict, train_logs: Dict[str, List[float]],
                  useCkpt: bool = False) -> Optional[List[ndarray]]:
        print('---> [7]: Bit2Edge: Finished Learning --> Make Prediction & Save on All Feature Set.')
        sleep(1e-3)
        if not BestValidationModel or not TRAIN_CALLBACKS['Model Checkpoint']:
            return None

        # Loading the weight first only to speed up the process
        if not self.__OpBestModel__(weight_mode=True, train_logs=train_logs, useCkpt=useCkpt) and \
                not self.__OpBestModel__(weight_mode=False, train_logs=train_logs, useCkpt=useCkpt):
            return None
        CODE = '-Ckpt' if useCkpt else ''
        MSG: Tuple[str, ...] = GetDtypeOfData('feature_set')
        KEY: Dict[str, str] = \
            {
                MSG[0]: f'{self._StorageFolder}Training Set [Pred{CODE}].csv',
                MSG[1]: f'{self._StorageFolder}Validation Set [Pred{CODE}].csv',
                MSG[2]: f'{self._StorageFolder}Testing Set [Pred{CODE}].csv',
                f'{MSG[1]}-{MSG[2]}': f'{self._StorageFolder}Dev-Test Set [Pred{CODE}].csv',
                f'{MSG[2]}-{MSG[1]}': f'{self._StorageFolder}Test-Dev Set [Pred{CODE}].csv',
            }
        return self._CorePredict_(DATASET=KEY, model=self.TF_Model, data_cache=data_cache)

    def _Evaluation_(self, FinalModel: str, data_cache: Dict) -> Optional[List[ndarray]]:
        print('---> [6]: Bit2Edge: Evaluation State --> Test Saving: Make Prediction & Save on All Feature Set.')
        tf.keras.backend.clear_session()

        PATH: str = FixPath(f'{self._StorageFolder}{FinalModel}', extension='.h5')
        print('Re-check:', PATH)

        SAVED_MODEL = B2E_Model(dataset=self.Dataset(), TF_Model=PATH, ModelKey=self.TF_Model.ModelKey)
        SAVED_MODEL.GetModel().set_weights(weights=self.TF_Model.GetModel().get_weights())

        MSG: Tuple[str, ...] = GetDtypeOfData('feature_set')
        KEY: Dict[str, str] = \
            {
                MSG[0]: f'{self._StorageFolder}Training Set [Pred-Recall on Final-Model].csv',
                MSG[1]: f'{self._StorageFolder}Validation Set [Pred-Recall on Final-Model].csv',
                MSG[2]: f'{self._StorageFolder}Testing Set [Pred-Recall on Final-Model].csv',
                f'{MSG[1]}-{MSG[2]}': f'{self._StorageFolder}Dev-Test Set [Pred-Recall on Final-Model].csv',
                f'{MSG[2]}-{MSG[1]}': f'{self._StorageFolder}Test-Dev Set [Pred-Recall on Final-Model].csv',
            }
        return self._CorePredict_(DATASET=KEY, model=SAVED_MODEL, data_cache=data_cache)

    def _CorePredict_(self, DATASET: Dict[str, str], model: B2E_Model, data_cache: Dict,
                      mode: int = 2) -> List[ndarray]:
        STATUS: Tuple[bool, ...] = \
            (
                self.Dataset().GetDataBlock('Info').GetData(environment='Train') is not None,
                self.Dataset().GetDataBlock('Info').GetData(environment='Val') is not None,
                self.Dataset().GetDataBlock('Info').GetData(environment='Test') is not None,
            )
        MSG: Tuple[str, ...] = GetDtypeOfData('feature_set')

        prediction: List[Optional[ndarray]] = []
        dfs: List[Optional[pd.DataFrame]] = []

        def GetTemplatePerfColumnForOneOutput() -> List[str]:
            return ['Target', 'Predict', 'RelError', 'AbsError']

        def GetTargetPropertiesType(num_output: int) -> List[str]:
            NOTIONS = MODEL_STRUCTURE.get('Predict-Notion', None)
            TestState(NOTIONS is not None, 'The prediction notion is not found, please check the config file.')
            TestState(len(NOTIONS) == len(set(NOTIONS)),
                      msg='The prediction notion is not unique, please check the config file.')
            if num_output == 1:
                idx = MODEL_STRUCTURE.get('Predict-Notion-Index', 0)
                notions = [NOTIONS[idx], ]  # 'a' notions -> len(GetTemplatePerfColumnForOneOutput) *'a' columns
            else:
                TestState(num_output <= len(NOTIONS), 'The number of output is exceeded the model representation.')
                notions = NOTIONS[:num_output]
            return notions

        for key, value in DATASET.items():
            try:
                idx = MSG.index(key)
            except ValueError:
                prediction.append(None)
                dfs.append(None)
                continue
            if not STATUS[idx]:
                prediction.append(None)
                dfs.append(None)
                continue

            if key not in data_cache:
                EnvData = self.Dataset().GetDataBlock('EnvData').GetData(environment=key)
                LBIData = self.Dataset().GetDataBlock('LBIData').GetData(environment=key)
                DATA = (EnvData, LBIData)
                F_Data, R_Data = model.GetDataV2(data=DATA, mode=mode, toTensor=True)
                data_cache[key] = (F_Data, R_Data)
            else:
                F_Data, R_Data = data_cache[key]

            y_pred: ndarray = model.PredictV2(fdata=F_Data, rdata=R_Data, mode=mode)
            prediction.append(y_pred)

            DataFrame = self.Dataset().InfoToDataFrame(request=key, Info=True, Environment=True, Target=False)
            dfs.append(DataFrame)

            target = self.Dataset().GetDataBlock('Target').GetData(environment=key)

            template = GetTemplatePerfColumnForOneOutput()
            tags = GetTargetPropertiesType(num_output=target.shape[1])
            for idx in range(0, target.shape[1]):
                tag = tags[idx]
                target_name, pred_name = f'{tag}: {template[0]}', f'{tag}: {template[1]}'
                DataFrame[target_name] = target[:, idx]
                DataFrame[pred_name] = y_pred[:, idx]
                DataFrame[f'{tag}: {template[2]}'] = DataFrame[target_name] - DataFrame[pred_name]
                DataFrame[f'{tag}: {template[3]}'] = DataFrame[f'{tag}: {template[2]}'].abs()

            ExportFile(DataFrame=DataFrame, FilePath=value)

        if dfs[1] is not None and dfs[2] is not None:
            if 'Val-Test' in DATASET or 'Dev-Test' in DATASET:
                ExportFile(pd.concat([dfs[1], dfs[2]], axis=0, ignore_index=True),
                           FilePath=DATASET['Val-Test'] if 'Val-Test' in DATASET else DATASET['Dev-Test'])
            if 'Test-Val' in DATASET or 'Test-Dev' in DATASET:
                ExportFile(pd.concat([dfs[2], dfs[1]], axis=0, ignore_index=True),
                           FilePath=DATASET['Test-Val'] if 'Test-Val' in DATASET else DATASET['Test-Dev'])

        return prediction

    def _Sketch_(self, title: str, y_pred: List[ndarray]) -> None:
        y_train = self.Dataset().GetDataBlock('Target').GetData(environment='Train')
        y_val = self.Dataset().GetDataBlock('Target').GetData(environment='Val')
        y_test = self.Dataset().GetDataBlock('Target').GetData(environment='Test')

        if y_train.ndim == 2:
            NUMS_TARGET: int = y_train.shape[1]
        else:
            NUMS_TARGET: int = 1

        def GetTarget(x1: ndarray, x2: ndarray, index: int) -> Tuple[ndarray, ndarray]:
            if x1 is None or x2 is None:
                if x1 is None and x2 is None:
                    raise ValueError('Two empty arrays.')
                if x1 is None:
                    warning('NO reference target can be found.')
                    res = x2[:, index] if x2.ndim == 2 else x2
                else:
                    warning('NO prediction can be found.')
                    res = x1[:, index] if x1.ndim == 2 else x1
                return res, res

            arr1 = x1[:, index] if x1.ndim == 2 else x1
            arr2 = x2[:, index] if x2.ndim == 2 else x2
            return arr1, arr2

        for idx in range(0, NUMS_TARGET):
            plt.clf()
            # plt.scatter(*GetTarget(y_train, y_train, index=idx), c='red', label='Train', alpha=0.75)
            y, x = GetTarget(y_train, y_pred[0], index=idx)
            plt.scatter(y, x, c='red', label='Train', alpha=0.75)

            if y_val is not None:
                y, x = GetTarget(y_val, y_pred[1], index=idx)
                plt.scatter(y, x, c='purple', label='Validation', alpha=0.75)

            if y_test is not None:
                y, x = GetTarget(y_test, y_pred[2], index=idx)
                plt.scatter(y, x, c='blue', label='Test', alpha=0.75)

            plt.plot([-30, 160], [-30, 160], linestyle='-.')
            _ShowPlot(xlabel='True Value (kcal/mol)', ylabel='ML Value (kcal/mol)', title=title)

        return None

    # Getter & Setter ------------------------------------------------------------------------------------------------
    def GetState(self) -> Dict:
        return self._ModelState

    def SaveModelConfig(self, FilePath: str, StorageFolder: Optional[str] = None) -> None:
        from Bit2Edge.config.devConfig import LR_LOGS, INIT_LR
        from Bit2Edge.config.modelConfig import SaveModelConfig, MODEL_STRUCTURE
        from Bit2Edge.config.startupConfig import COMPUTING_DEVICE

        ModelFramework = FRAMEWORK.copy()
        if isinstance(self.TF_Model, B2E_Model):
            ModelFramework['ModelKey'] = self.TF_Model.ModelKey
        ModelFramework['TRAIN_CALLBACKS'] = TRAIN_CALLBACKS
        ModelFramework['INIT_LR'] = INIT_LR
        ModelFramework['LR_LOGS'] = LR_LOGS
        ModelFramework['MODEL_STRUCTURE'] = MODEL_STRUCTURE
        ModelFramework['COMPUTING_DEVICE'] = COMPUTING_DEVICE

        if StorageFolder is None:
            StorageFolder = self._StorageFolder
        if StorageFolder != '':
            StorageFolder = FixPath(StorageFolder, extension='/')
        SaveModelConfig(FilePath=StorageFolder + FilePath, ModelConfig=ModelFramework)
