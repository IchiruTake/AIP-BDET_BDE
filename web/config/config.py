# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
from functools import lru_cache
from logging import warning

from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.dataObject.FileParseParams import FileParseParams
from Bit2Edge.deploy.deploy import DEPLOYMENT
from Bit2Edge.input.BondParams import BondParams
from Bit2Edge.input.Creator import FeatureEngineer
from Bit2Edge.test.GroupTester import GroupTester
from Bit2Edge.test.params.MolDrawParams import MolDrawParams
from Bit2Edge.test.params.PredictParams import PredictParams
from Bit2Edge.utils.file_io import ReadFile
from Bit2Edge.utils.helper import GetIndexOnArrangedData


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
@lru_cache
def _LoadDrawParams() -> MolDrawParams:
    DrawParams = MolDrawParams()
    DrawParams.DenyMolImageFile = True
    DrawParams.ImageSize = (250, 250)
    DrawParams.SortOnTarget = False
    return DrawParams


def _PreloadFile(filename: str, params: FileParseParams) -> dict:
    """
        This is used to store the basic mapping of smiles and a series of bond_index and its result
        into dictionary. The format is :
        {'smiles':
            {
                'index': [(start_index, end_index), ...],  # <-- The start and end index of finding the molecule
                'bond_index': {
                    'bond_result_column': 'bond_result'
                    ...
                    '_list': 'bond_results_list'
                }
            }
        }
        Parameters:
        ----------

        filename : string
            The filepath to compute the mapping of bond_index and its result into dictionary

        params : FileParseParams
            The file structure to load


        Returns:
        -------
            A dictionary mapping bond_index and its result on each smiles
    """

    # Load file
    data, cols = ReadFile(filename, header=0, get_values=True, get_columns=True)
    index = GetIndexOnArrangedData(array=data, cols=params.Mol(), get_last=True)

    bIdx = data[:, params.BondIndex()].tolist()
    targets = data[:, params.Target()].tolist()

    config = {}
    for i in range(len(index) - 1):
        # Setup default
        start_index: int = index[i][0]
        next_index: int = index[i + 1][0]
        smiles: str = index[i][1]
        if smiles not in config:
            config[smiles] = {
                'index': [],
            }
        temp = config[smiles]
        temp['index'].append((start_index, next_index))

        # Load bIdx and targets (No error handling, use latest value)
        for j in range(start_index, next_index):
            bond_index = bIdx[j]
            if not isinstance(bond_index, int):
                bond_index = int(bond_index)
            if bond_index not in temp:
                temp[bond_index] = {}
            else:
                warning(f'The bond index {bond_index} in {smiles} at row {j} has been collected once (duplicated).')
            for k, col_id in enumerate(params.Target()):
                temp[bond_index][cols[col_id]] = targets[j][k]
            temp[bond_index]['_list'] = targets[j]

    return config


DEPLOYMENT_CONFIG: dict = {
    'IS_LOADED': False,
    'MODEL_SEED': 1,
    'MODEL_NAME': None,
    'MODEL_VERBOSE': True,
    'TRAIN_FILE': {
        'FILENAME': 'BDE_data/source_dataset_v1.csv',
        'FILE_PARAMS': FileParseParams(mol=0, radical=(1, 2), bIdx=3, bType=4, target=[5]),
        'RESULT_CONFIG': None,
        'SCOPE': {'C', 'H', 'O', 'N'},
    },
    'DRAW_PARAMS': _LoadDrawParams(),
    'PRED_PARAMS': PredictParams(),
    'FILE_PARAMS': FileParseParams(mol=0, radical=(1, 2), bIdx=3, bType=4, target=None),
    'BOND_PARAMS': BondParams(),
    'NULL_VALUE': -9999.0,
    'DEFAULT_MOL_MODE': 'SMILES',
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    'DATASET': None,
    'GENERATOR': None,
    'MODEL_PLACEHOLDER': None,
    'TESTER': None,
}


def IsDeploymentConfigLoaded():
    return DEPLOYMENT_CONFIG.get('IS_LOADED', False) is True


def OnStartup() -> None:
    if IsDeploymentConfigLoaded():  # Stop additional loading
        return None

    settings = DEPLOYMENT_CONFIG

    # [1]: Load the train file
    filename: str = settings['TRAIN_FILE']['FILENAME']
    file_params: FileParseParams = settings['TRAIN_FILE']['FILE_PARAMS']
    settings['TRAIN_FILE']['RESULT_CONFIG'] = _PreloadFile(filename=filename, params=file_params)

    # [2]: Load the backend engine
    settings['DATASET'] = FeatureData(trainable=False, retrainable=False)
    settings['GENERATOR'] = FeatureEngineer(dataset=settings['DATASET'])

    # [3]: Initialize ModelPlaceholder and Tester
    SEED = settings['MODEL_SEED']
    NAME = settings['MODEL_NAME']
    VERBOSE = settings['MODEL_VERBOSE']

    settings['MODEL_PLACEHOLDER'] = DEPLOYMENT.MakeGroupPlaceholder(seed=SEED, name=NAME)
    # _ = settings['MODEL_PLACEHOLDER'].ReloadInputState()
    settings['TESTER'] = GroupTester(dataset=settings['DATASET'], generator=settings['GENERATOR'],
                                     placeholder=settings['MODEL_PLACEHOLDER'])

    # [4]: Marked as done
    settings['IS_LOADED'] = True
    return None


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# [2.2]: Train-File-Config
def GetDeploymentVariable(*args):
    result = DEPLOYMENT_CONFIG
    for idx, arg in enumerate(args):
        result = result.get(arg, None)
        if result is None or not (isinstance(result, dict) and idx != len(args) - 1):
            break
    return result
