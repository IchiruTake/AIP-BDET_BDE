# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from logging import warning
from typing import Optional, Tuple, Union

from Bit2Edge.dataObject.DataBlock import GetDtypeOfData
from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.utils.verify import TestState, TestStateByWarning

_DTYPE = GetDtypeOfData()
_INFO = _DTYPE['info']
_FEATURE = _DTYPE['feature']
_LABEL = _DTYPE['label']
_TARGET = _DTYPE['target']
# --------------------------------------------------------------------------------


class DatasetLinker:
    __slots__ = ('_dataset', '_feature_set_key', '_isActive', '__KEY__', '_roll')

    def __init__(self, dataset: FeatureData):
        TestState(dataset is not None, 'The object::data should not be None.')
        TestState(isinstance(dataset, FeatureData), f'The object::data should be the instance of {FeatureData} class.')
        self._dataset: Optional[FeatureData] = None
        self._feature_set_key: Optional[str] = None # << Current key, can be rotated when call RollToNextKey
        self.__KEY__: Optional[Tuple[bool, bool]] = None
        self._roll: bool = False
        self.BindNewDataset(dataset=dataset)

    def BindNewDataset(self, dataset: FeatureData) -> None:
        TestStateByWarning(self._dataset is None or dataset is self._dataset,
                           "We don't recommend this method, but the attribute `dataset` has been changed.")
        TestState(isinstance(dataset, FeatureData), f'The :arg:`dataset` should be the instance of {FeatureData}.')
        self._dataset: FeatureData = dataset
        self._feature_set_key: str = dataset.OptFeatureSetKey()
        self.__KEY__: Tuple[bool, bool] = (dataset.trainable, dataset.retrainable)
        self._roll: bool = False

    # [1]: Getter | Setter Function =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def GetDataInBlock(self, request: str, env_key: Optional[str] = None) -> Union[_INFO, _TARGET, _FEATURE]:
        env_key = env_key or self._feature_set_key  # << Should I use self.GetKey() instead of self._feature_set_key?
        return self.Dataset().GetDataBlock(request).GetData(environment=env_key)

    def SetDataInBlock(self, data, request: str, env_key: Optional[str] = None) -> None:
        env_key = env_key or self._feature_set_key  # << Should I use self.GetKey() instead of self._feature_set_key?
        self.Dataset().GetDataBlock(request).SetData(data, environment=env_key)

    def GetTargetReference(self):
        return self.GetDataInBlock(request='Target')

    # [2]: Other Function ---------------------------------------------------------------------------------------------
    def DetachDataset(self) -> FeatureData:
        warning('This implementation should not be called as deleting dataset is dangerous.')
        DATASET = self._dataset
        self._dataset = None
        return DATASET

    def Dataset(self) -> FeatureData:
        return self._dataset

    def GetKey(self) -> str:
        if not self._roll:  # If the key is allowed to be rotated, then we don't do this check
            dataset = self.Dataset()
            TestState(self.__KEY__ == (dataset.trainable, dataset.retrainable),
                      msg='The state of the dataset is now vulnerable.')
        return self._feature_set_key

    def GetFixedKey(self) -> str:
        return self.Dataset().OptFeatureSetKey()

    def RollToNextFeatureSetKey(self) -> str:
        self._roll = True
        SetMsg: Tuple[str, ...] = GetDtypeOfData('feature_set')
        old_index: int = SetMsg.index(self._feature_set_key)
        new_index: int = (old_index + 1) % len(SetMsg)
        warning(f'The key has been migrated from :{self._feature_set_key}: to :{SetMsg[new_index]}:.')
        self._feature_set_key = SetMsg[new_index]
        return self.GetKey()

    def ResetKey(self) -> str:
        self._roll = False
        self._feature_set_key = self.GetFixedKey()
        warning(f'The key has been reset back to :{self._feature_set_key}:.')
        return self.GetKey()
