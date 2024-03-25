# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module served as an intermediate placeholder that ignored the setting of
# :arg:`TF_Model`
# --------------------------------------------------------------------------------

from typing import Dict, List, Any, Optional

from Bit2Edge.config.userConfig import UpdateDataConfig, DATA_FRAMEWORK, UpdateDataConfigFromDict
from Bit2Edge.dataObject.InputState import InputState
from Bit2Edge.input.BVManager import BVManager
from Bit2Edge.utils.file_io import ReadLabelFile


class BasePlaceholder(object):

    def __init__(self, ModelConfig: Dict[str, str], name: str = None):
        self._name: str = name
        self._state_: dict = ModelConfig.copy()
        self._cache: Dict[str, Any] = {}

    # --------------------------------------------------------------------------------
    def GetName(self) -> str:
        return self._name

    def ReloadInputState(self, copy: bool = True) -> Dict:
        if self._cache.get('SavedInputConfig', None) is not None:
            DATA_FRAMEWORK.clear()
            DATA_FRAMEWORK.update(self._cache['SavedInputConfig'])
            return self._cache['SavedInputConfig']

        UpdateDataConfig(self.GetSavedInputConfig(), RemoveOldRecord=True)
        InputState.ResetInputState()
        if self._cache.get('SavedInputConfig', None) is None:
            self._cache['SavedInputConfig'] = DATA_FRAMEWORK.copy()

        return self._cache['SavedInputConfig'] if copy else DATA_FRAMEWORK

    def _SetupEnvFilePath(self) -> None:
        if self._cache.get('EnvFilePath', None) is None:
            self._cache['EnvFilePath'] = ReadLabelFile(FilePath=self.GetEnvFilePath(), header=0)
        return None

    def _SetupLBIFilePath(self) -> None:
        if self._cache.get('LBIFilePath', None) is None:
            self._cache['LBIFilePath'] = ReadLabelFile(FilePath=self.GetLBIFilePath(), header=0)
        return None

    def _SetupInputConfig(self) -> None:
        if self._cache.get('SavedInputConfig', None) is None or self._cache.get('BVManager', None) is None:
            CurrentDataFramework = DATA_FRAMEWORK.copy()

            UpdateDataConfig(self.GetSavedInputConfig(), RemoveOldRecord=True)
            InputState.ResetInputState()
            self._cache['SavedInputConfig'] = DATA_FRAMEWORK.copy()
            self._cache['BVManager']: BVManager = BVManager(stereochemistry=False)  # Stereo-chemistry can be any.

            UpdateDataConfigFromDict(dictionary=CurrentDataFramework, RemoveOldRecord=True)
        return None

    def Setup(self) -> None:
        self._SetupEnvFilePath()
        self._SetupLBIFilePath()
        self._SetupInputConfig()

    # --------------------------------------------------------------------------------
    def GetTFModelPath(self) -> Optional[str]:
        return self._state_.get('TF_Model', None)

    def GetEnvFilePath(self) -> Optional[str]:
        return self._state_.get('EnvFilePath', None)

    def GetDecodedEnvFilePath(self) -> List[str]:
        self._SetupEnvFilePath()
        return self._cache['EnvFilePath']

    def GetLBIFilePath(self) -> Optional[str]:
        return self._state_.get('LBIFilePath', None)

    def GetDecodedLBIFilePath(self) -> List[str]:
        self._SetupLBIFilePath()
        return self._cache['LBIFilePath']

    def GetModelKey(self) -> Optional[str]:
        return self._state_.get('ModelKey', None)

    def GetSavedInputConfig(self) -> Optional[str]:
        return self._state_.get('SavedInputConfig', None)

    def GetDecodedSavedInputConfig(self) -> Dict[str, Any]:
        self._SetupInputConfig()
        return self._cache['SavedInputConfig']

    def GetDecodedBVManager(self) -> Optional[BVManager]:
        return self._cache.get('BVManager', None)

    def GetSavedModelConfig(self) -> str:
        return self._state_['SavedModelConfig']

    def GetState(self) -> Dict[str, Any]:
        return self._state_
