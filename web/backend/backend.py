# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from flask import Markup

import web.config.config as CONF
import web.backend.validation as BKE_VAL
from Bit2Edge.config.startupConfig import GetPrebuiltInfoLabels
from Bit2Edge.dataObject.DataBlock import DEFAULT_OUTPUT_NPDTYPE
from Bit2Edge.test.GroupTester import GroupTester
from Bit2Edge.test.TargetDefinition import TargetDefinition
from Bit2Edge.test.TesterUtilsP1 import CastTarget
from web.config.config import IsDeploymentConfigLoaded, OnStartup
from cachetools import cached, TTLCache
from time import perf_counter


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def _LoadTargetInTrainScope(smiles: str, BondIndexList: List[int]) -> np.ndarray:
    TRAIN_FILE_PARAMS = CONF.GetDeploymentVariable('TRAIN_FILE', 'FILE_PARAMS')
    NULL_VALUE = CONF.GetDeploymentVariable('NULL_VALUE')

    def GetOneUnkRecord():
        return [NULL_VALUE] * len(TRAIN_FILE_PARAMS.Target())

    # [1]: Check if smiles is seen in training set
    TRAIN_MOL_PROFILE = BKE_VAL.LoadMolInTrainScope(smiles)
    if TRAIN_MOL_PROFILE is None:
        return np.array([GetOneUnkRecord() for _ in range(len(BondIndexList))], dtype=DEFAULT_OUTPUT_NPDTYPE)

    # [2]: If found, load all the target found
    return np.array([TRAIN_MOL_PROFILE[bond_index]['_list'] if bond_index in TRAIN_MOL_PROFILE else GetOneUnkRecord()
                     for i, bond_index in enumerate(BondIndexList)], dtype=DEFAULT_OUTPUT_NPDTYPE)


def _GetReferenceLabels():
    labeling = TargetDefinition.GetPredictionLabels
    TRAIN_FILE_PARAMS = CONF.GetDeploymentVariable('TRAIN_FILE', 'FILE_PARAMS')
    return labeling(num_output_each=len(TRAIN_FILE_PARAMS.Target()), num_model=1, term='ref')


def _SetTrainTargetProfileInDf(df: pd.DataFrame) -> pd.DataFrame:
    if not IsDeploymentConfigLoaded():
        raise RuntimeError('The deployment configuration is not loaded.')
    TESTER = CONF.GetDeploymentVariable('TESTER')

    # [1]: Load the smiles and bond index
    params = TESTER.GetParams()
    smiles: str = str(df[df.columns[params.Mol()]].values[0])
    BondIndexList = df[df.columns[params.BondIndex()]].values.tolist()

    # [2]: Load the profile
    result = _LoadTargetInTrainScope(smiles=smiles, BondIndexList=BondIndexList)
    labels = _GetReferenceLabels()
    for idx, column in enumerate(labels):
        df[column] = result[:, idx]

    return df


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
@cached(TTLCache(maxsize=2048, ttl=600, timer=perf_counter), lock=None)
def _PredictSmiles(smiles: str, mode: str = 'SMILES') -> pd.DataFrame:
    # [1]: Validate input molecule, the smiles here is forced to be correct
    smiles = BKE_VAL.GetCanonicalSmiles(smiles=smiles, mode=mode, ignore_error=False)
    tester: GroupTester = CONF.GetDeploymentVariable('TESTER')

    # [2]: Load molecule and run feature engineering
    p0 = CONF.GetDeploymentVariable('BOND_PARAMS')
    tester.AddMol(mol=smiles, mode=mode, canonicalize=True, useChiral=True, params=p0)
    tester.CreateData(BitVectState=None, LBondInfoState=True)

    # [3]: Make prediction
    p1 = CONF.GetDeploymentVariable('PRED_PARAMS')
    tester.predict(params=p1)

    # [4]: Draw molecule using the "ensemble" prediction to highlight
    p2 = CONF.GetDeploymentVariable('DRAW_PARAMS')
    RefMode: int = len(tester.GetCache()['performance-label']) - 3  # "ensemble" prediction index
    df: pd.DataFrame = tester.DrawMol(FolderPath='', RefMode=RefMode, params=p2)

    # [5]: Cleanup
    params = CONF.GetDeploymentVariable('FILE_PARAMS')
    df = df.sort_values(by=[df.columns[params.BondIndex()]])
    df = _SetTrainTargetProfileInDf(df=df)
    if 'svg' in df:
        df['svg'] = df['svg'].apply(Markup)
    # DisplayDataFrame(df)
    return df


def _CastDfForRender(df: pd.DataFrame) -> List:
    tester = CONF.GetDeploymentVariable('TESTER')

    # [1]: Obtain labels
    t_labels: List[str] = tester.GetCache()['prediction-label']
    t_codes: List[Dict[str, str]] = [TargetDefinition.DecodeLabel(t_label) for t_label in t_labels]
    print('render of tester:', t_labels, t_codes)

    r_labels: List[str] = _GetReferenceLabels()
    r_codes: List[Dict[str, str]] = [TargetDefinition.DecodeLabel(r_label) for r_label in r_labels]
    print('render of reference:', r_labels, r_codes)
    PREBUILT_INFO_LABELS = GetPrebuiltInfoLabels()

    # [2]: Get params
    DRAW_PARAMS = CONF.GetDeploymentVariable('DRAW_PARAMS')
    PRED_PARAMS = CONF.GetDeploymentVariable('PRED_PARAMS')
    NULL_VALUE = CONF.GetDeploymentVariable('NULL_VALUE')

    ResultDisplay = []
    NumSingleModel = tester.GetNumSingleModels()
    for key, value in df.iterrows():
        # [3.1]: Load easy variables (SVG, Bond Index, Bond Type)
        temp = [value['svg'], value[PREBUILT_INFO_LABELS[3]], value[PREBUILT_INFO_LABELS[4]], [], []]

        # [3.2]: Load all result from one prediction (maybe not supported for multi-output)
        for idx, (code, label) in enumerate(zip(t_codes, t_labels)):
            BaseValue = [code['notion'].upper(), None, CastTarget(value[label], Sfs=DRAW_PARAMS.Sfs)]
            if isinstance(tester, GroupTester):
                BaseValue[1] = f'ML #{idx + 1}'
                if idx == NumSingleModel and (PRED_PARAMS.average or PRED_PARAMS.ensemble):
                    BaseValue[1] = 'ML-Average' if PRED_PARAMS.average else 'ML-Ensemble'
                if idx == NumSingleModel + 1 and PRED_PARAMS.ensemble:
                    BaseValue[1] = 'ML-Ensemble'
            else:
                BaseValue[1] = 'ML'
            temp[3].append(BaseValue)

        # [3.2]: Load all results from the trained reference (maybe not supported for multi-output)
        for idx, (code, label) in enumerate(zip(r_codes, r_labels)):
            if value[label] > NULL_VALUE + 1:
                BaseValue = [code['notion'].upper(), 'DFT', CastTarget(value[label], Sfs=DRAW_PARAMS.Sfs)]
                temp[4].append(BaseValue)

        ResultDisplay.append(temp)
    return ResultDisplay


def _CastDfForRenderV2(df: pd.DataFrame) -> List[dict]:
    tester = CONF.GetDeploymentVariable('TESTER')

    # [1]: Obtain labels
    t_labels: List[str] = tester.GetCache()['prediction-label']
    t_codes: List[Dict[str, str]] = [TargetDefinition.DecodeLabel(t_label) for t_label in t_labels]

    r_labels: List[str] = _GetReferenceLabels()
    r_codes: List[Dict[str, str]] = [TargetDefinition.DecodeLabel(r_label) for r_label in r_labels]
    PREBUILT_INFO_LABELS = GetPrebuiltInfoLabels()

    # [2]: Get params
    DRAW_PARAMS = CONF.GetDeploymentVariable('DRAW_PARAMS')
    PRED_PARAMS = CONF.GetDeploymentVariable('PRED_PARAMS')
    NULL_VALUE = CONF.GetDeploymentVariable('NULL_VALUE')

    reports = []
    NumSingleModel = tester.GetNumSingleModels()
    for key, value in df.iterrows():
        # [3.1]: Load easy variables (SVG, Bond Index, Bond Type)
        temp = {
            'smiles': value[PREBUILT_INFO_LABELS[0]],
            'bond_index': value[PREBUILT_INFO_LABELS[3]],
            'bond_type': value[PREBUILT_INFO_LABELS[4]],
            'svg': value['svg'],
            'ML': [],
            'DFT/Exp': [],
        }

        # [3.2]: Load all result from the trained reference
        for idx, (code, label) in enumerate(zip(r_codes, r_labels)):
            if value[label] > NULL_VALUE + 1:
                report = {
                    'notion': code['notion'].upper(),
                    'model': 'DFT',
                    'value': CastTarget(value[label], Sfs=DRAW_PARAMS.Sfs),
                }
                temp['DFT/Exp'].append(report)

        # [3.3]: Load all result from one prediction (maybe not supported for multi-output)
        for idx, (code, label) in enumerate(zip(t_codes, t_labels)):
            report = {
                'notion': code['notion'].upper(),
                'model': f'#{idx + 1}',
                'value': CastTarget(value[label], Sfs=DRAW_PARAMS.Sfs),
            }
            if isinstance(tester, GroupTester):
                if idx == NumSingleModel and (PRED_PARAMS.average or PRED_PARAMS.ensemble):
                    report['model'] = 'Avg' if PRED_PARAMS.average else 'Ens'
                if idx == NumSingleModel + 1 and PRED_PARAMS.ensemble:
                    report['model'] = 'Ens'

            temp['ML'].append(report)

        reports.append(temp)
    return reports

@cached(TTLCache(maxsize=2048, ttl=600, timer=perf_counter), lock=None)
def _FinalizePrediction(smiles: str, mode: str = CONF.GetDeploymentVariable('DEFAULT_MOL_MODE')) \
        -> Tuple[pd.DataFrame, List[dict]]:
    df = _PredictSmiles(smiles=smiles, mode=mode)
    report = _CastDfForRenderV2(df)
    return df, report

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def RunPrediction(smiles: str, mode: str = 'SMILES') -> dict:
    if not IsDeploymentConfigLoaded():
        OnStartup()

    result = {}

    # [1]: Make prediction and convert to simpler configuration
    df, report = _FinalizePrediction(smiles=smiles, mode=mode)
    result['df'] = df
    result['report'] = report

    # [2]: Is molecule in training profile?
    temp = (BKE_VAL.LoadMolInTrainScope(smiles=smiles, mode=mode) is not None)
    result['is_trained'] = temp

    # [3]: Evaluate atomic state?
    temp = BKE_VAL.EvalAtomInMol(smiles=smiles, train=True, non_train=True)
    result['atomic_state'] = temp

    # [4]: Message returned on this prediction
    if len(result['atomic_state']['non_train']) != 0:
        non_trained_atoms = list(result['atomic_state']['non_train'].keys())
        msg = (f'Warning: SMILES = "{smiles}" is out of the training scope ({non_trained_atoms}) '
               f'which can have lower confidence as usual.')
        status = 'Out-of-Range'
    else:
        base_term: str = f'SMILES = "{smiles}" is successfully predicted'
        if not result['is_trained']:
            msg = f'Info: {base_term} but lower confidence can be met.'
        else:
            msg = f'Info: {base_term} with high confidence.'
        status = 'OK'
    result['message'] = {
        'text': msg,
        'type': msg.split(':')[0].upper(),
        'status': status,
    }

    return result
