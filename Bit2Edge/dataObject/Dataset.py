# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from logging import info
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import cached_property
import numpy as np
import pandas as pd
from numpy import ndarray

from Bit2Edge.dataObject.DataBlock import DataBlock, GetDtypeOfData, FIXED_INPUT_NPDTYPE, DEFAULT_OUTPUT_NPDTYPE
from Bit2Edge.dataObject.FileParseParams import FileParseParams
from Bit2Edge.dataObject.InputState import InputState
from Bit2Edge.input.FeatureUtils import EvalFingerprintLabels
from Bit2Edge.utils.cleaning import (DeleteDataByMask, GetLocationForLabelRemoval, ComputeMaskForFeatureCleaning,
                                     RunGarbageCollection)
from Bit2Edge.utils.file_io import ExportFile, FixPath, ReadLabelFile, TestIsValidFilePath
from Bit2Edge.utils.verify import InputFullCheck, TestState, TestStateByWarning, TestStateByInfo

_DTYPE = GetDtypeOfData()
_INFO = _DTYPE['info']
_FEATURE = _DTYPE['feature']
_LABEL = _DTYPE['label']
_TARGET = _DTYPE['target']


# --------------------------------------------------------------------------------


class FeatureData:
    """
    The following class is similar as the DataLoader which would contain and manage the data, supported
    data division (train/val/test) in DatasetSplitter.

    """

    _AttributeSets_: Tuple[str, ...] = ('Info', 'Target', 'Env', 'EnvData', 'LBIData')

    # __slots__ = ('_datablock', 'ThisLabelInfo', '__trainable', '__retrainable', '__params', '_state_',
    #              'trainable', 'retrainable', 'scratchtrain', 'pretrain', 'testable')

    def __init__(self, trainable: bool, retrainable: bool = False):
        # [1.1a]: Basic Information
        self._datablock: Dict[str, DataBlock] = {}
        for AttributeSet in FeatureData._AttributeSets_:
            self._datablock[AttributeSet] = DataBlock(name=AttributeSet)

        # [1.2]: Information needs to thorough
        # Follow-up Features: Generator should be equal to CFile and SFile
        self.ThisLabelInfo: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None

        # [2]: CountingLabels Information
        # [2.1]: Fixed Information
        TestState(trainable or not retrainable, msg=':arg:`retrainable` is invalid due to :arg:`trainable` = True.')
        self.__trainable: bool = trainable
        self.__retrainable: bool = retrainable

        # [2.2]: Feature Location at InfoData
        self.__params = FileParseParams(mol=0, radical=(1, 2), bIdx=3, bType=4, target=None)

        # [3]: Extra attributes
        self._state_: Dict[str, Union[Dict, Any]] = {}

    # [2]: Advanced Modification -------------------------------------------------------------------------------------
    def EvalEnvLabelInfo(self) -> None:
        labels = self.GetDataBlock('EnvData').GetColumns()
        self.ThisLabelInfo = EvalFingerprintLabels(labels=labels)

    def GetEnvLabelInfo(self, update: bool = False) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        if update or (self.ThisLabelInfo is None and self.GetDataBlock('EnvData').GetColumns() is not None):
            self.EvalEnvLabelInfo()
        return self.ThisLabelInfo

    # [3]: Generator Function ----------------------------------------------------------------------------------------
    def _TestCleaningRequirement_(self, EnvFilePath: Optional[str], request: str) -> None:
        """
        This function is a part of :meth:`CleanEnvData()` to ensure all the requirements are met. Assuming
        the state of this object (:class:`FeatureData`) is corrected without possibly memory overflow,
        the idea as follows:

        1. If state=ScratchTraining => EnvFilePath can be either None or a file passed.
        2. If state=PreTraining => EnvFilePath must NOT be None => A file must be passed.
        3. If state=Testing => EnvFilePath must NOT be None => A file must be passed.

        Another thing we may need to test is the availability of the data/ndarray stored.
        1. If state=ScratchTraining or PreTraining => Training set must NOT be None.
        2. If state=Testing => Testing set must NOT be None.

        This is to ensure the :class:`DatasetLinker` whose :arg:`key`='auto' can be correctly
        identified and not vulnerable.
        """

        if self.scratchtrain:
            TestState(EnvFilePath is None or TestIsValidFilePath(EnvFilePath),
                      msg=f'The :arg:`EnvFilePath`={EnvFilePath} is invalid.')
        elif self.pretrain or self.testable:
            TestState(EnvFilePath is not None and TestIsValidFilePath(EnvFilePath),
                      msg=f'The :arg:`EnvFilePath`={EnvFilePath} is invalid.')
        else:
            raise ValueError('This state is invalid.')

        TestState(self.GetDataBlock('Info').GetData(environment=request) is not None,
                  msg=f'{request}-ing set is invalid.')
        TestState(self.GetDataBlock('EnvData').GetData(environment=request) is not None,
                  msg=f'{request}-ing set is invalid.')

    def CleanEnvData(self, EnvFilePath: Optional[str], request: Optional[str] = 'Train') -> None:
        """
        This method implements Data Cleaning only if this object was set as 'train'. If there is
        pre-built model with saved label(s), it can be recalled to skip feature-searching.

        Arguments:
        ---------
        
        EnvFilePath : str
            If set, this file will be used for Data Cleaning.
        
        request : str 
            This is to set the dataset chosen to remove the feature. Default to be `train`. If not 
            specified, it used the default key by :meth:`self.OptKey()`.

        """
        # [0]: Hyper-parameter Verification
        if request is None or request.lower() in ('auto', 'default'):
            request: str = self.OptFeatureSetKey()
        self._TestCleaningRequirement_(EnvFilePath=EnvFilePath, request=request)
        TestState(request in GetDtypeOfData('feature_set'),
                  msg=f':arg:`request`={request} is invalid. Please try again.')
        TestStateByWarning(request == self.OptFeatureSetKey(),
                           f'The current request={request} is NOT compatible with the '
                           f'current state of this object ({self.OptFeatureSetKey()})')
        FEATURE_TYPES = [request] + [rq for rq in GetDtypeOfData('feature_set') if rq != request]
        blockname = 'EnvData'

        SET_1 = self.GetDataBlock(blockname).GetData(environment=FEATURE_TYPES[0])
        labels = self.GetDataBlock(blockname).GetColumns()
        preShape: Tuple[int, int] = SET_1.shape

        src_data = [self.GetDataBlock(blockname).GetData(environment=feature_type)
                    for feature_type in FEATURE_TYPES if feature_type != FEATURE_TYPES[0]]
        duration: float = perf_counter()
        print('Data Cleaning is on processed. Please hold on for a secs ...')
        if EnvFilePath is None:
            # forceClean must be False to disable cleaning
            MASK: List[int] = ComputeMaskForFeatureCleaning(SET_1, labels, *src_data)
        else:
            TargetLabels: List[str] = ReadLabelFile(FilePath=EnvFilePath, header=0)
            # print(f'CurrentLabels: {len(labels)}')
            # print(f'TargetLabels: {len(TargetLabels)}')
            MASK: List = GetLocationForLabelRemoval(SourceLabels=labels, TargetLabels=TargetLabels)

        if len(MASK) != 0:
            print('Please waiting as features are requested to be deleted.')
            timer: float = perf_counter()
            SET_1, src_data, labels = DeleteDataByMask(MASK, SET_1, labels, *src_data)
            RunGarbageCollection(0)
            self.GetDataBlock(blockname).SetColumns(labels)
            self.GetDataBlock(blockname).SetData(SET_1, environment=FEATURE_TYPES[0])
            for i, data in enumerate(src_data, start=1):
                if data is not None:
                    self.GetDataBlock(blockname).SetData(data, environment=FEATURE_TYPES[i])

            timer = perf_counter() - timer
            print(f"Calling array deletion: {timer:.4f} (s) -> {1e3 * timer / preShape[1]:.4f} (ms/feature).")
        else:
            info('No feature is removed.')

        START, END = self.GetEnvLabelInfo(update=(len(MASK) != 0))
        duration: float = perf_counter() - duration
        print(f'After Cleaning: StartPtrs: {START} --- EndPtrs: {END}')
        print(f"Executing Time for Data Cleaning: {duration:.4f} (s) "
              f"-> {1e3 * duration / preShape[1]:.4f} (ms/feature).")
        print(
            f"Modified Shape: {preShape} -> {self.GetDataBlock(blockname).GetData(environment=FEATURE_TYPES[0]).shape}.")
        print(f"Modified Labels: ({preShape[1]}, ) -> {self.GetDataBlock(blockname=blockname).GetColumns().shape}.")
        return None

    # [4]: Export Information ---------------------------------------------------------------------------------------
    def InfoToDataFrame(self, request: str, Info: bool = True, Target: bool = True,
                        Environment: bool = False) -> pd.DataFrame:
        """
        This function is to do the DataFrame conversion for the information data, including the 
        InfoData, TargetData, and EnvironmentData. The default setting is to export all the
        information data (but not the EnvironmentData). 

        Arguments:
        ---------

        request: str
            The request of the data. It should be one of the following: 'Train', 'Val', 'Test'.
        
        Info: bool
            The flag to export the InfoData. Default is True.
        
        Target: bool
            The flag to export the TargetData. Default is True.
        
        Environment: bool
            The flag to export the EnvironmentData. Default is False.

        Returns:
        -------
        DataFrame: pd.DataFrame
            The DataFrame of the information data.
        """
        max_uint8: int = np.iinfo(FIXED_INPUT_NPDTYPE).max
        InfoData, EnvData, TargetData = self.GetDataInformation(environment=request)
        InfoLabels, EnvLabels, TargetLabels = self.GetLabelInformation()

        if Info is True and (InfoData is not None and InfoLabels is not None):
            df: pd.DataFrame = pd.DataFrame(data=InfoData, index=None, columns=InfoLabels)
            PARAMS = self.GetParams()
            maxValue = FIXED_INPUT_NPDTYPE if df[InfoLabels[PARAMS.BondIndex()]].max() < max_uint8 else np.uint32
            df[InfoLabels[PARAMS.BondIndex()]] = df[InfoLabels[PARAMS.BondIndex()]].astype(maxValue)
        else:
            df: pd.DataFrame = pd.DataFrame()

        if Environment is True and (EnvData is not None and EnvLabels is not None):
            for _, (label, data) in enumerate(zip(EnvLabels, EnvData.T)):
                df[label] = data

            maxValue = FIXED_INPUT_NPDTYPE if df[EnvLabels[-1]].max() < max_uint8 else np.uint32
            df[EnvLabels[-1]] = df[EnvLabels[-1]].astype(maxValue)

        if Target is True and (TargetData is not None and TargetLabels is not None):
            for _, (label, data) in enumerate(zip(TargetLabels, TargetData.T)):
                df[label] = data

        return df

    def ExportInfoToCsv(self, request: str, FileName: str, Info: bool = True, Target: bool = True,
                        Environment: bool = False) -> pd.DataFrame:
        """
        This function is an extension of the FeatureData.InfoToDataFrame() function. It is to export the
        pd.DataFrame of the information data to the csv file. The default setting is to export all the
        information data (but not the EnvironmentData).

        Arguments:
        ---------

        request: str
            The request of the data. It should be one of the following: 'Train', 'Val', 'Test'.
        
        FileName: str
            The file name of the exported csv file.
        
        Info: bool
            The flag to export the InfoData. Default is True.
        
        Target: bool
            The flag to export the TargetData. Default is True.
        
        Environment: bool
            The flag to export the EnvironmentData. Default is False.
        
        Returns:
        -------
        DataFrame: pd.DataFrame
            The DataFrame of the information data.
        """

        # [1]: Initialization
        SetMsg = GetDtypeOfData('feature_set')
        KEY: Dict[str, str] = \
            {
                SetMsg[0]: 'Training File',
                SetMsg[1]: 'Validation File',
                SetMsg[2]: 'Testing File',
            }

        # [2]: Export the information data
        FileName = FixPath(FileName, extension='.csv')
        print(f'{KEY[request]} >> {FileName}.')
        DF = self.InfoToDataFrame(request, Info=Info, Target=Target, Environment=Environment)
        ExportFile(DataFrame=DF, FilePath=FileName, index=False, index_label=None)
        return DF

    def ExportFeatureLabels(self, Env_FileName: str, LBI_FileName: str) -> None:
        """
        This function is to export the feature labels of the bit-vect feature and the local bond
        information feature.

        Arguments:
        ---------
        Env_FileName: str
            The file name of the exported feature labels of the bit-vect feature. If None, then
            the feature labels of the bit-vect feature will not be exported.
        
        LBI_FileName: str
            The file name of the exported feature labels of the local bond information feature. If
            None, then the feature labels of the local bond information feature will not be exported.
        
        """
        storage = (('EnvData', Env_FileName), ('LBIData', LBI_FileName))
        for blockname, filename in storage:
            if filename is not None:
                label = self.GetDataBlock(blockname).GetColumns()
                ExportFile(DataFrame=pd.DataFrame(data=None, index=None, columns=label),
                           FilePath=FixPath(FileName=Env_FileName, extension='.csv'))
        return None

    # [4]: Basic Modification ----------------------------------------------------------------------------------------
    # [4.1]: Basic Getter & Setter
    @staticmethod
    def GetDataBlockName() -> Tuple[str, ...]:
        return FeatureData._AttributeSets_

    def _GetDatablock(self, blockname: str, error: bool = False) -> Optional[DataBlock]:
        msg = f'[Warning] Request is invalid, must be one of these {FeatureData._AttributeSets_}.'
        if error is True:
            msg = msg.replace('[Warning]', '[Error]')
            TestState(blockname in FeatureData._AttributeSets_, msg=msg)
        else:
            TestStateByWarning(blockname in FeatureData._AttributeSets_, msg=msg)
        return self._datablock.get(blockname, None)

    def GetDataBlock(self, blockname: str) -> Optional[DataBlock]:
        return self._GetDatablock(blockname, error=True)

    def GetDataInformation(self, environment: str) -> Tuple[Union[ndarray, Any], ...]:
        INFO = self.GetDataBlock(FeatureData._AttributeSets_[0]).GetData(environment=environment)
        ENV = self.GetDataBlock(FeatureData._AttributeSets_[2]).GetData(environment=environment)
        TARGET = self.GetDataBlock(FeatureData._AttributeSets_[1]).GetData(environment=environment)
        return INFO, ENV, TARGET

    def GetLabelInformation(self) -> Tuple[_LABEL, ...]:
        INFO = self.GetDataBlock(blockname=FeatureData._AttributeSets_[0]).GetColumns()
        ENV = self.GetDataBlock(blockname=FeatureData._AttributeSets_[2]).GetColumns()
        TARGET = self.GetDataBlock(blockname=FeatureData._AttributeSets_[1]).GetColumns()
        return INFO, ENV, TARGET

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # [4.3]: Dataset State: Train/Test
    @cached_property
    def trainable(self) -> bool:
        return self.__trainable

    @cached_property
    def retrainable(self) -> bool:
        return self.__retrainable

    @cached_property
    def scratchtrain(self) -> bool:
        return self.__trainable and not self.__retrainable

    @cached_property
    def pretrain(self) -> bool:
        return self.__trainable and self.__retrainable

    @cached_property
    def testable(self) -> bool:
        return not self.__trainable

    def OptFeatureSetKey(self) -> str:
        SearchMessage: Tuple[str, ...] = GetDtypeOfData('feature_set')
        return SearchMessage[0] if self.trainable else SearchMessage[2]

    def GetParams(self) -> FileParseParams:
        return self.__params

    def GetState(self) -> Dict[str, Any]:
        return self._state_

    # []: Advanced Function
    def RefreshData(self) -> None:
        """ This method will reset your data. """
        self._state_.clear()
        RunGarbageCollection(1)
        for blockname in FeatureData._AttributeSets_:
            if blockname in ('Info', 'Target', 'Env'):
                datablock = self.GetDataBlock(blockname=blockname)
                datablock.ClearData()
        self.RefreshFeatureOnly()
        self.ThisLabelInfo = None

    def RefreshFeatureOnly(self) -> None:
        for blockname in FeatureData._AttributeSets_:
            if blockname in ('EnvData', 'LBIData'):
                datablock = self.GetDataBlock(blockname=blockname)
                datablock.ClearData()
        return None
