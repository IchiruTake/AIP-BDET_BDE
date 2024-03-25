# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
from typing import Union, Optional

from Bit2Edge.test.placeholder.PlaceholderUtils import LoadConfiguration
from Bit2Edge.deploy.deployCode import SAVED_MODEL_AT_EPOCH, SAVED_MODEL_WEIGHTS
from Bit2Edge.test.placeholder.GroupPlaceholder import GroupPlaceholder
from Bit2Edge.test.placeholder.SinglePlaceholderV1 import SinglePlaceholderV1
from Bit2Edge.test.placeholder.SinglePlaceholderV2 import SinglePlaceholderV2

# --------------------------------------------------------------------------------
class DEPLOYMENT:
    BaseFilePath: str = 'model'
    SeedCode: str = 'S'
    RunCode: str = 'R'
    ModelTagSuffix: Optional[str] = None
    ModelFile: str = 'TF-Keras Checkpoint'

    EnvFile: str = 'Env_SavedLabel [After].csv'
    LBIFile: str = 'LBI_SavedLabel [After].csv'
    SavedInputConfigFile: str = 'Data Configuration [After].yaml'
    SavedModelConfigFile: str = 'Model Configuration [After].yaml'

    @staticmethod
    def _GetFilePath(seed: int) -> str:
        return DEPLOYMENT.BaseFilePath + f'/{DEPLOYMENT.SeedCode}{seed}'

    @staticmethod
    def GetModelFilename(seed: int, run: int, epoch_hint: Optional[int] = None) -> str:
        EPOCH: str = str(SAVED_MODEL_AT_EPOCH[seed][run] if epoch_hint is None else epoch_hint)
        if len(EPOCH) < 3:
            EPOCH = '0' * (3 - len(EPOCH)) + EPOCH
        BaseName = DEPLOYMENT._GetFilePath(seed=seed)
        BaseName = BaseName + f'/{DEPLOYMENT.RunCode}{run}' + f'/{DEPLOYMENT.ModelFile} {EPOCH}'
        if DEPLOYMENT.ModelTagSuffix is not None:
            BaseName = BaseName + f' [{DEPLOYMENT.ModelTagSuffix}]'
        return BaseName + '.h5'

    # --------------------------------------------------------------------------------
    @staticmethod
    def MakeSinglePlaceholder(seed: int, run: int, ver: int = 2, name: Optional[str] = None,
                              epoch_hint: Optional[int] = None) -> Union[SinglePlaceholderV1, SinglePlaceholderV2]:
        if epoch_hint is not None:
            if ver != 1:
                raise ValueError('If :arg:`epoch_hint` is specified, the :arg:`version` must be 1')
        BasePath = DEPLOYMENT._GetFilePath(seed=seed)
        TempConfig = LoadConfiguration(EnvFilePath=BasePath + f'/{DEPLOYMENT.EnvFile}',
                                       LBIFilePath=BasePath + f'/{DEPLOYMENT.LBIFile}',
                                       SavedInputConfigFile=BasePath + f'/{DEPLOYMENT.SavedInputConfigFile}',
                                       SavedModelConfigFile=BasePath + f'/{DEPLOYMENT.SavedModelConfigFile}',
                                       TF_Model=DEPLOYMENT.GetModelFilename(seed=seed, run=run, epoch_hint=epoch_hint))
        NAME: str = name or f'V{ver}-S{seed}-R{run}'
        if ver == 1:
            pl = SinglePlaceholderV1(TempConfig, name=NAME)
        else:
            pl = SinglePlaceholderV2(TempConfig, name=NAME)
            pl.SetWeights(SAVED_MODEL_WEIGHTS[seed][1][run])

        pl.ReloadInputState(copy=True)
        return pl

    @staticmethod
    def MakeGroupPlaceholder(seed: int, name: Optional[str] = None) -> GroupPlaceholder:
        group = GroupPlaceholder(name=name or f'Group-S{seed}')
        for idx, run in enumerate([*SAVED_MODEL_AT_EPOCH[seed]]):
            placeholder = DEPLOYMENT.MakeSinglePlaceholder(seed=seed, run=run, ver=2, name=None)
            if idx == 0:
                placeholder.ReloadInputState(copy=True)
            group.Register(placeholder)

        group.SetWeights(SAVED_MODEL_WEIGHTS[seed][0])
        return group


# --------------------------------------------------------------------------------
def _GetFilePath(seed: int) -> str:
    return DEPLOYMENT.BaseFilePath + f'/{DEPLOYMENT.SeedCode}{seed}'


def GetModelName(seed: int, run: int, ModelEpoch: Optional[int] = None) -> str:
    ModelCode: str = str(SAVED_MODEL_AT_EPOCH[seed][run] if ModelEpoch is None else ModelEpoch)
    if len(ModelCode) < 3:
        ModelCode = '0' * (3 - len(ModelCode)) + ModelCode
    Params = DEPLOYMENT
    BaseName = _GetFilePath(seed=seed) + f'/{Params.RunCode}{run}' + f'/{Params.ModelFile} {ModelCode}'
    if Params.ModelTagSuffix is not None:
        BaseName = BaseName + f' [{Params.ModelTagSuffix}]'
    return BaseName + '.h5'
