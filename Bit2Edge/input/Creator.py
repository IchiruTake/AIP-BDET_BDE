# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

import gc
import time
from logging import warning
from time import perf_counter, sleep
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from rdkit.Chem.rdchem import Mol

from Bit2Edge.config.startupConfig import GetPrebuiltInfoLabels
from Bit2Edge.config.userConfig import DATA_FRAMEWORK as dFramework, SaveDataConfig
from Bit2Edge.dataObject.DataBlock import GetDtypeOfData, FIXED_INPUT_NPDTYPE, FIXED_INFO_NPDTYPE, \
    DEFAULT_OUTPUT_NPDTYPE
from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.dataObject.DatasetLinker import DatasetLinker
from Bit2Edge.dataObject.FileParseParams import FileParseParams
from Bit2Edge.dataObject.InputState import InputState
from Bit2Edge.input.BVManager import BVManager
from Bit2Edge.molUtils.bitVectUtils import StringVectToNumpy
from Bit2Edge.input.CreatorUtils import _ConstraintDataset_
from Bit2Edge.input.EnvExtractor import EnvExtractor
from Bit2Edge.input.LBondInfo import LBICreator
from Bit2Edge.molUtils.molUtils import PyCacheEdgeConnectivity, SmilesFromMol, CanonMolString, SmilesToSanitizedMol
from Bit2Edge.utils.cleaning import RunGarbageCollection
from Bit2Edge.utils.helper import GetIndexOnArrangedData, ReadFile, Sort, KeySortParams
from Bit2Edge.utils.verify import InputFastCheck, InputFullCheck, TestState, TestStateByWarning, TestStateByInfo

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Doing bit-vect translation
_bvengine = BVManager(stereochemistry=dFramework['FP-StereoChemistry'])

def _BitVectTask(mols) -> ndarray:
    return StringVectToNumpy(_bvengine.GetBitVectAPI(mols))


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class FeatureEngineer(DatasetLinker):
    """
    This class is a wrapper that handle three specific components, which are:
    1) EnvExtractor.EnvExtractor: Convert the ((kekulized) molecule and bond to a list of sub-mols
        Caller: Method GetEnvAPI() -> Tuple[Mol, ...] and bondIdx
    
    2) BVManager.BVManager: Convert sub-mols in (1) to a series of fingerprint
        Caller: Method GetBitVectAPI() -> A bit-vect Python string
    
    3) LBondInfo.LBICreator: Extract the properties of the central bond using the sanitized molecule
        Caller: Method GetLBondInfoAPI() -> A list of properties (by integer).
    
    This class is the only component on the main we want to work with, zero additional code add-up 
    in three components. 
    """

    BaseFpLabels: Optional[Tuple[str, ...]] = None

    __slots__ = ('_params', '_state', '_safe_mode',
                 '_EnvExtractor', '_BVManager', '_LBICreator',
                 '_GetEnvsAPI_', '_GetBitVectAPI_', '_GetLBondInfoAPI_',
                 '_InfoData', '_EnvData', '_LBondInfoData', '_Environment', '_FeatureLocation',
                 '_MolTable', '_BondIdxList')

    def __init__(self, dataset: FeatureData = None, storeEnvironment: bool = False,
                 showInvalidStereo: bool = False, TIMER: bool = False, verbose_mol_check: bool = False,
                 process: Union[int, str] = 1, max_process: int = 3):
        InputFullCheck(process, name='process', dtype='int-str', delimiter='-')
        InputFullCheck(max_process, name='max_process', dtype='int')
        if isinstance(process, str):
            process = process.lower()
            if process not in ('auto', ):
                raise ValueError('The :arg:`process` argument must be either "auto" or and integer.')
        else:
            TestState(1 <= process <= max_process, msg='The :arg:`process` must be smaller than the :arg:`max_process`.')
        super(FeatureEngineer, self).__init__(dataset=dataset)

        # -----------------------------------------------------------------------------------
        # [1]: Dataset Pointer & Initialization
        # [1.1]: Data Position ---> Cache Purpose Only
        self._params = FileParseParams(mol=0, radical=(1, 2), bIdx=3, bType=4, target=None)

        # [1.2]: Initialization
        self._state = {
            'storeEnvironment': storeEnvironment,
            'showInvalidStereochemistry': showInvalidStereo,
            'verbose_mol_check': verbose_mol_check,

            # 'BV_Reduce' = any(self._state['BitVect'])),
            'LBondInfo': True, 'BitVect': (True,) * len(InputState.GetNames()), 'BV_Reduce': True,

            'create_bde': 0, 'convert_bde': 0, 'create_time': 0.0, 'convert_time': 0.0,
            '_TIMER': TIMER, 'env_time': 0.0, 'fp_time': 0.0, 'lbi_time': 0.0,
            'process': process, 'max_proc': max_process,
        }

        # [2]: Extract all components
        self._EnvExtractor: EnvExtractor = EnvExtractor(RecordBondIdxOn2SmallEnv=self._state['storeEnvironment'])
        self._BVManager: BVManager = BVManager(stereochemistry=dFramework['FP-StereoChemistry'])
        self._LBICreator: LBICreator = LBICreator(mode=None, StereoChemistry=dFramework['LBI-StereoChemistry'])

        # [2.2]: Extra State
        self._GetEnvsAPI_: Callable = self._EnvExtractor.GetEnvsAPI
        self._GetBitVectAPI_: Callable = self._BVManager.GetBitVectAPI
        self._GetLBondInfoAPI_: Callable = self._LBICreator.GetLBondInfoAPI

        # [3]: Working with data
        # [3.1]: Internal-Built Cache
        # An attribute determine the 'row' and the first SMILES in an arranged data
        self._MolTable: Optional[List[Tuple[int, str]]] = None
        self._BondIdxList: Optional[List[int]] = None  # A list of bond index in the dataset

        # [3.2]: Pointer Cache of self.GetDataset() & its state
        self._InfoData: Optional[ndarray] = None
        self._EnvData: Optional[ndarray] = None
        self._LBondInfoData: Optional[ndarray] = None
        self._Environment: Optional[ndarray] = None

        # This attribute indicated the location of each bit-vect group's location
        self._FeatureLocation: Tuple = self._BVManager.GetFeatureLocation(mergeOption=True)
        self._safe_mode: bool = False

    # ------------------------------------------------------------------------------------------------------------
    # Step 01: Add data into Feature Data. See Step 02 below.
    def _ValidateAddData_(self, data: ndarray, column: Union[List[str], ndarray], params: FileParseParams,
                          sorting: bool) -> None:
        TestState(self._InfoData is None, 'The data has been pre-initialized. Please check your codeflow.')
        TestState(len(column) == data.shape[1],
                  'The labels and data are incompatible. Please check your codeflow.')
        InputFullCheck(sorting, name='sorting', dtype='bool')
        params.IsEnoughComponent(target=False, error=True)

    def AddData(self, data: ndarray, column: Union[List[str], ndarray], params: FileParseParams,
                sorting: bool = False, limitTest: Union[int, float, List, Tuple] = None, ) -> None:
        # [0]: Hyper-parameter Verification
        self._CacheDataset_()
        self._ValidateAddData_(data=data, column=column, params=params, sorting=sorting)

        print('-' * 33, 'Adding Data', '-' * 34)
        # [1]: Re-build Dataset
        params.TestParams(maxSize=data.shape[1])
        if limitTest is not None:
            data = _ConstraintDataset_(data, limitTest=limitTest)

        if sorting:
            request: str = input('Do you want to sort the data by the first column? (Y/N): ')
            key = KeySortParams(status=request in ('Y', 'Yes', 'y'), column=params.Mol(), key=str,
                                reverse=False, maxSize=data.shape[1])
            task = KeySortParams(status=True, column=params.BondIndex(), key=int,
                                 reverse=False, maxSize=data.shape[1])
            data = Sort(data, KeyParams=key, SortColParams=task)

        # [2]: Put Data Into Consideration
        stack: List[int] = params.DistributeData(useTarget=False)
        self.SetDataInBlock(np.array(data[:, stack], dtype=np.object_), request='Info')
        try:
            self.Dataset().GetDataBlock('Info').SetColumns(np.array(GetPrebuiltInfoLabels(), dtype=np.object_).tolist())
        except ValueError:
            self.Dataset().GetDataBlock('Info').SetColumns(np.array(column, dtype=np.object_)[stack].tolist())

        if params.Target() is not None:
            target = params.Target()
            self.SetDataInBlock(np.array(data[:, target], dtype=DEFAULT_OUTPUT_NPDTYPE), request='Target')
            self.Dataset().GetDataBlock('Target').SetColumns(np.array([column[idx] for idx in target], dtype=np.object_))

        return self._CacheDataset_()

    def AddCsvData(self, FilePath: str, params: FileParseParams, sorting: bool = False,
                   ascending: Tuple[bool, bool] = (True, True),
                   limitTest: Union[int, float, List, Tuple] = None, debug: bool = True) -> None:
        # [1]: Read File With Requests
        InputFullCheck(debug, name='debug', dtype='bool')

        print('-' * 29, 'Adding Data with CSV', '-' * 29)
        File: pd.DataFrame = ReadFile(FilePath, header=0, usecols=params.DistributeData(useTarget=True))

        # [2]: Processing your file
        if sorting:
            if InputFastCheck(ascending, dtype='bool'):
                ascending = (ascending, ascending)
            else:
                TestState(len(ascending) == 2, f'Two required values for ascending={ascending} only.')
                InputFullCheck(ascending[0], name='ascending[0]', dtype='bool')
                InputFullCheck(ascending[1], name='ascending[1]', dtype='bool')
            File.sort_values(by=[File.columns[self._params.Mol()], File.columns[self._params.BondIndex()]],
                             ascending=ascending, inplace=True)

        self.AddData(data=File.values, column=File.columns.tolist(), params=params,
                     sorting=False, limitTest=limitTest)

        # [3]: Display the state. This also implied hidden validation on `self.AddData()` to ensure
        # the data is ready for train / test.
        print('-' * 30, '!!! YOUR FILE !!!', '-' * 31)
        if debug:
            msg: str = 'Error source code at func::AddData().'
            TestState(self.GetDataInBlock(request='Info') is not None, msg=msg)
            TestState(self.Dataset().GetDataBlock('Info').GetColumns() is not None, msg=msg)
            if params.Target() is not None:
                TestState(self.GetDataInBlock(request='Target') is not None, msg=msg)
                TestState(self.Dataset().GetDataBlock('Target').GetColumns() is not None, msg=msg)
        File.info(memory_usage='deep')
        del File
        RunGarbageCollection(1)

    # -------------------------------------------------------------------------------------------------------------
    # Step 03: Feature Engineering
    def GetDataAPIs(self, GC: bool = True, SAFE: bool = False, SortOnValTest: bool = False) -> None:
        print('-' * 120)
        timer: float = perf_counter()
        SetMsg: Tuple[str, ...] = GetDtypeOfData('feature_set')
        for key in SetMsg:
            # [1]: Clean the data
            print(f'Current Key: {key}.')
            self.RefreshData()
            # [2]: Run feature engineering
            InfoData = self.GetDataInBlock(request='Info')
            if InfoData is not None:
                if SortOnValTest and key in (SetMsg[1], SetMsg[2]):
                    print(f'The key: {key} is running sorting.')
                    data = {
                        'mol': InfoData[:, self._params.Mol()],
                        'bIdx': InfoData[:, self._params.BondIndex()]
                    }
                    df = pd.DataFrame(data).sort_values(['mol', 'bIdx'], inplace=False, ascending=[True, True])
                    INDEX = df.index.values.tolist()

                    super(FeatureEngineer, self).SetDataInBlock(InfoData[INDEX, :], request='Info')
                    target_data = self.GetDataInBlock(request='Target')
                    if target_data is not None:
                        super(FeatureEngineer, self).SetDataInBlock(target_data[INDEX, :], request='Target')

                self.GetDataAPI(GC=GC, SAFE=SAFE)

            # [3]: Move to next key
            super(FeatureEngineer, self).RollToNextFeatureSetKey()
            time.sleep(0.01)
        super(FeatureEngineer, self).ResetKey()
        print(f"Full Execution time for Generator from {timer:.4f} (s): {perf_counter() - timer:.4f} (s).")
        print('-' * 120)

    def GetDataAPI(self, GC: bool = True, SAFE: bool = False) -> None:
        """
        This method would be run after method `_PrepareData()` and is used to operate the pipeline. Note that this
        function required the input to be consistently verified by the SMILES-BondIdx order. The evaluation has done
        here is very loosely and lazily so the user are asked to input them correctly

        Parameters:
        ----------

        - GC : bool
            Whether to enable the Python garbage collector. Default to True.

        - SAFE : bool
            If set to True, the algorithm behaved correctly on any universal graph where the connectivity between two
            vertices are represented by one edge only. If :arg:``SAFE``=False, it would cache the molecular graph to
            speed up the algorithm. Disable this can improve the performance, and up to now no issue occured. Default
            to False.

        """
        # [1]: Initialization
        if True:
            InputFullCheck(GC, name='GC', dtype='bool')
            InputFullCheck(SAFE, name='safe', dtype='bool')

            TestState(self.Dataset() is not None, 'The dataset pointer is None')
            self._CacheDataset_()
            TestState(self._InfoData is not None, 'No data has been allocated. Please check your codeflow.')

            # A check to see if the duplication when training is correct when using FeatureData.DuplicateDataset().
            TargetData: Optional[ndarray] = self.GetDataInBlock(request='Target')
            if self._InfoData is not None and TargetData is not None:
                x: int = self._InfoData.shape[0]
                y: int = TargetData.shape[0]
                TestState(x == y, f'The InfoData (nSamples={x}) and TargetData (nSamples={y}) is size incompatible.')
            pass

        # [2]: Generate the data
        print('-' * 20, 'Generate Features', '-' * 21)
        self._PrepareData()

        if self._state['BV_Reduce'] or self._state['LBondInfo']:
            from rdkit import __version__ as RD_VERSION
            from rdkit.RDLogger import DisableLog, EnableLog
            # [1]: Generate Features
            # Since 2022.03, the function RemoveHs will initialize the logger. This logger is not available on
            # previous version, and only available when calling :meth:`RemoveHs` to store the SMILES environment
            # See here: https://github.com/rdkit/rdkit/issues/2683
            # This is called at :meth:`CheckInvalidStereoChemistry()`
            RDK_VERSION = RD_VERSION.split('.')

            if not GC and gc.isenabled():
                gc.disable()
            if int(RDK_VERSION[0]) >= 2022:
                DisableLog('rdApp.*')

            self._safe_mode = SAFE
            self._DataWrapper_(begin=0, stop=None)

            if not gc.isenabled():
                gc.enable()
            if int(RDK_VERSION[0]) >= 2022:
                EnableLog('rdApp.*')

            self.ReportSpeed()

        label: ndarray = self.Dataset().GetDataBlock('EnvData').GetColumns()
        print(f'Current LBondInfoData: {self._LBondInfoData.shape} ---> DataType: {self._LBondInfoData.dtype}.')
        print(f'Current EnvData: {self._EnvData.shape} ---> DataType: {self._EnvData.dtype}.')
        print(f'Current Labels: {label.shape} ---> DataType: {label.dtype}.')

        # [3]: [Optional] Feature Deactivation for Measuring Feature Importance
        totalTime: float = perf_counter()
        # This is used for test process only
        if not self.Dataset().trainable and not all(self._state['BitVect']):
            # Remove the left-over fingerprint: when use BondEnv-4 (inactive) to create LBondInfo (active).
            if not self._state['BitVect'][-1]:  # Possible Radical-Env or Last Bond-Env
                t1 = InputState.GetStartLocationIfPossibleRadicalEnv()
                t2 = InputState.GetNumsInput() - 1
                self._EnvData[:, self._FeatureLocation[t1][0]:self._FeatureLocation[t2][1]] = 0

            for idx, activation in enumerate(self._state['BitVect']):  # Bond-Env
                if activation is False and idx != len(self._state['BitVect']) - 1:
                    self._EnvData[:, slice(*self._FeatureLocation[idx], 1)] = 0

        timer: float = self._state['create_time'] + self._state['convert_time'] + (perf_counter() - totalTime)
        NSamples: int = self._InfoData.shape[0]
        print(f"Executing Time for Generator: {timer:.4f} (s) --> {1e3 * timer / NSamples:.4f} (ms / bond).")
        print('-' * 60)

    def _DataWrapper_(self, begin: int = 0, stop: Optional[int] = None, ) -> None:
        """
            This function is the core function where we generate and control all features needed
            in a single pipeline, which is just a caller wrapper for three components. This
            function is called during method `self.GetDataAPI()`.
        """
        # [0]: Preparing Data
        self._MolTable = GetIndexOnArrangedData(self._InfoData, cols=self._params.Mol(), get_last=True)
        BondIdxList = self._InfoData[:, self._params.BondIndex()].tolist()
        for idx, value in enumerate(BondIdxList):
            if not isinstance(value, int):
                BondIdxList[idx] = int(value)
        self._BondIdxList = BondIdxList
        stop: int = len(self._MolTable) - 1 if stop is None else stop

        # [1]: Constructing StateTable to generate data for each sample points at minimal cost
        StateTable = self._BuildCache_(begin=begin, stop=stop)

        # [2]: If the sample points are not seen previously, generate that samples.
        # Otherwise, cast it from available resources.
        self._GenerateTask_(BuildList=StateTable['MERGE'])
        RunGarbageCollection(0)
        self._PassData_(forth=StateTable[True], reverse=StateTable[False])
        self._BondIdxList = None

    def _PassData_(self, forth: List[int], reverse: List[int]):
        # [3]: Passing similar features --> All Features needed to be passed is passed (Both Forward and Reversed)
        IS_STORE_ENVIRONMENT: bool = self._state['storeEnvironment']
        beginTime: float = perf_counter()

        for CORE_ROW, CACHE_ROW in forth:
            # Copy data on a row, not reference
            self._EnvData[CACHE_ROW, :] = self._EnvData[CORE_ROW, :]
            self._LBondInfoData[CACHE_ROW, :] = self._LBondInfoData[CORE_ROW, :]
            if IS_STORE_ENVIRONMENT:
                self._Environment[CACHE_ROW, :] = self._Environment[CORE_ROW, :]

        for CORE_ROW, CACHE_ROW in reverse: # As I know, the reverse is not needed, but kept for refactoring
            self._EnvData[CACHE_ROW, :] = self._EnvData[CORE_ROW, :]
            self._LBondInfoData[CACHE_ROW, :] = self._LBondInfoData[CORE_ROW, :]
            if IS_STORE_ENVIRONMENT:
                self._Environment[CACHE_ROW, :] = self._Environment[CORE_ROW, :]

        self._state['convert_time']: float = perf_counter() - beginTime

    # --------------------------------------------------
    # Step 03.0: Feature Engineering Utility
    @staticmethod
    def _ToIterProgress_(BuildList: List, msg: str = 'Feature Generator Progress: ') -> Any:
        """ This function received a list of row-id integer to indicate which bond
            should be generated. """
        sleep(0.01)
        try:
            from tqdm import tqdm
            return tqdm(BuildList, ncols=75 + len(msg), position=0, desc=msg, mininterval=0.5,
                        total=len(BuildList))
        except (ImportError, ImportWarning):
            warning('tqdm package is not available.')
        return BuildList

    def _DisableUnusedMolForBitVectOp_(self, mols: List[Mol]) -> Optional[List[Mol]]:
        # This operation is the optimization control over bit-vect during inference.
        if not self.Dataset().testable:
            return mols

        assert len(mols) == len(self._state['BitVect']), \
            'The number of molecules is not compatible with the :args:`self._state[BitVect]`. '

        return [mol if state or (idx == self._EnvExtractor.bIdxLocation) else None
                for idx, (mol, state) in enumerate(zip(mols, self._state['BitVect']))]

    def _StoreEnvMol_(self, mols: List[Mol], bondIdx: int, row: int) -> None:
        envs: List = [bondIdx] * (InputState.GetNumsInput() + 1)
        for i, mol in enumerate(mols):
            envs[i] = SmilesFromMol(mol, keepHs=True)
        self._Environment[row, :] = envs

    # --------------------------------------------------
    # Step 03.1: Prepare :arg:`MolTable` for Feature Engineering
    @staticmethod
    def _MatchReactionByRadicals_(r_base: List[str], r_check: List[str]) -> Optional[bool]:
        """ This function will compare the current reaction with the previous reaction to check similarity. """
        # None = Generate new data; True = Copy old data; False = Copy old data but form reversing.
        if r_check[0] == r_base[1] and r_check[1] == r_base[0]:
            return True # No support for RE-Learner anymore
        elif r_check[0] == r_base[0] and r_check[1] == r_base[1]:
            return True
        return None

    def _UpdateCache_(self, StateTable: Dict[Optional[bool], Union[ndarray, List[int]]]):
        if len(self._BondIdxList) != len(StateTable[None]) + len(StateTable[True]) + len(StateTable[False]):
            raise ValueError(f'The pipeline (_BuildCache*_) got error.')
        print('READY.')
        print(f'Generator: {len(StateTable[None])} <-~-> Switching: {len(StateTable[True])}-{len(StateTable[False])}.')

    def _BuildCache_(self, begin: int, stop: int) -> Dict:
        """ This method will create a simple cache to identify the reaction we needed to truly
            generate data and which are not. """
        print('Cache is built: ... -> ', end='')
        StateTable: Dict[Optional[Union[bool, str]], List[Union[int, Tuple[int, int], List[int]]]] = \
            {
                'MERGE': [],
                None: [],
                True: [],
                False: []
            }
        RadicalStack: List[List[str, str]] = self._InfoData[:, self._params.Radical()].tolist()
        RADICAL_MATCHING: Callable = FeatureEngineer._MatchReactionByRadicals_
        VERBOSE_MOL_CHECK: bool = self._state['verbose_mol_check']

        def _DetectIsomeric_(r_base: List[str], r_check: List[str]) -> bool:
            # This method is to validate the isomeric notation defined in the radical, if they have the same
            # bond index. Also, this can help to check if the reactions are correctly defined
            r_base_1 = CanonMolString(r_base[0], mode='SMILES', useChiral=False)
            r_base_2 = CanonMolString(r_base[1], mode='SMILES', useChiral=False)
            r_check_1 = CanonMolString(r_check[0], mode='SMILES', useChiral=False)
            r_check_2 = CanonMolString(r_check[1], mode='SMILES', useChiral=False)
            if r_base_1 == r_check_1 and r_base_2 == r_check_2:
                return True
            elif r_base_1 == r_check_2 and r_base_2 == r_check_1:
                return True
            return False

        for moleculeSet in range(begin, stop):  # range(0, len(self._MolTable) - 1)
            start, end = self._MolTable[moleculeSet][0], self._MolTable[moleculeSet + 1][0]
            mask: List[bool] = [False] * (end - start)

            # Doing identification
            for r1 in range(start, end):
                if mask[r1 - start]:
                    continue
                for r2 in range(r1 + 1, end):
                    if mask[r2 - start]:
                        continue
                    CheckRadicalMatching = RADICAL_MATCHING(r_base=RadicalStack[r1], r_check=RadicalStack[r2])
                    ShareSameBondIdx: bool = self._BondIdxList[r1] == self._BondIdxList[r2]
                    if isinstance(CheckRadicalMatching, bool) and ShareSameBondIdx:
                        self._state['convert_bde'] += 1
                        StateTable[CheckRadicalMatching].append((r1, r2))
                        mask[r2 - start] = True
                    elif CheckRadicalMatching is None and not ShareSameBondIdx:
                        pass
                    elif VERBOSE_MOL_CHECK:
                        # :meth:`RADICAL_MATCHING()` here cannot detect isomeric.
                        temp = _DetectIsomeric_(r_base=RadicalStack[r1], r_check=RadicalStack[r2])
                        # print('-' * 40)
                        mol = self._MolTable[moleculeSet][1]
                        bIdx: int = self._BondIdxList[r1]
                        if temp is True:
                            msg = f'\n[Info] An isomeric reactions has been duplicated at molecule = "{mol}" ' \
                                  f'with bond index = "{bIdx}". \nThe reactions are found at r1={r1} (base) and ' \
                                  f'r2={r2} (check). \n'
                        else:
                            msg = f'\n [Warn] The possible invalid reaction are found at molecule = "{mol}" ' \
                                  f'with bond index = "{bIdx}". \nThe reactions are found at r1={r1} (base) and ' \
                                  f'r2={r2} (check). \n'
                        msg += f'Base: {RadicalStack[r1][0]} ++ {RadicalStack[r1][1]}\n' \
                               f'Check: {RadicalStack[r2][0]} ++ {RadicalStack[r2][1]}'
                        (TestStateByWarning if temp else TestStateByInfo)(False, msg)
                        # print('-' * 40)

            # Initiate build result
            StateTable['MERGE'].append([])
            for row in range(start, end):
                if not mask[row - start]:
                    StateTable[None].append(row)
                    StateTable['MERGE'][-1].append(row)

        self._state['create_bde'] = self._InfoData.shape[0] - self._state['convert_bde']
        self._UpdateCache_(StateTable=StateTable)
        del RadicalStack
        return StateTable

    # --------------------------------------------------
    # Step 03.2: Execute Feature Engineering
    def _GenerateTask_(self, BuildList: List[List[int]]) -> None:
        """ This method is aimed to generate data each task """
        # [1.1]: Initialization
        timing: bool = self._state['_TIMER']
        RunBitVectOps: bool = self._state['BV_Reduce']
        RunLBIOps: bool = self._state['LBondInfo']

        NUM_PROCESSES: Union[int, str] = self._state['process']
        if NUM_PROCESSES == 'auto': # Heuristically determined
            NUM_PROCESSES = self._state['max_proc'] if self._EnvData.shape[0] > 20000 else 1
        else:
            TestState(NUM_PROCESSES >= -1 and NUM_PROCESSES != 0,
                      msg='[Error]: The number of processes is -1 or limited to the number of processes.')
            if NUM_PROCESSES == -1:
                NUM_PROCESSES = self._state['max_proc']
            NUM_PROCESSES = min(NUM_PROCESSES, self._state['max_proc']) # Code safety
        MULTI_PROC: bool = NUM_PROCESSES != 1

        # [1]: Running loop without multiprocessing along with LBI context and Env|BitVect context
        beginTime: float = perf_counter()
        BuildObject = self._ToIterProgress_(list(enumerate(BuildList)))
        cached_result = {'row': [], 'bv': []}
        safe = self._safe_mode
        for MolIndex, rows in BuildObject:
            # [1.1]: Obtained sanitized molecule
            smiles: str = self._MolTable[MolIndex][1]
            SMOL = SmilesToSanitizedMol(smiles)
            PyCacheEdgeConnectivity(SMOL)

            # [1.2]: Run LBI feature
            if RunBitVectOps:
                if not MULTI_PROC:
                    self._LazyBitVectOp_(rows, safe, SMOL, timing)
                else:
                    self._ParallelBitVectOp_(rows, safe, SMOL, timing, cached_result=cached_result)

            if RunLBIOps:
                self._LBondInfoOp_(rows, safe, SMOL, timing)

        # [2]: Running loop with multiprocessing along with Env|BitVect context
        if RunBitVectOps and MULTI_PROC:
            t0 = perf_counter() if timing else 0

            from multiprocessing import Pool
            with Pool(processes=NUM_PROCESSES) as p:
                bitvect = p.map(_BitVectTask, cached_result['bv'])
                for row, bv in zip(cached_result['row'], bitvect):
                    self._EnvData[row, :] = bv

            if timing:
                self._state['fp_time'] += perf_counter() - t0
            del cached_result

        self._state['create_time']: float = perf_counter() - beginTime

    def _LazyBitVectOp_(self, rows: List[int], safe: bool, SMol: Mol, timing: bool) -> None:
        # [1]: Preparation
        StoreEnv: bool = self._state['storeEnvironment']
        temp_BondIdxList = [self._BondIdxList[row] for row in rows]

        # [2]: EnvTime
        t0: float = perf_counter() if timing else 0
        p_row = []
        p_bitvect = []
        for i, (mols, bIdx) in enumerate(self._GetEnvsAPI_(SMol, temp_BondIdxList, safe)):
            row: int = rows[i]
            if StoreEnv:
                self._StoreEnvMol_(mols, bondIdx=bIdx, row=row)
            p_row.append(row)
            p_bitvect.append(self._DisableUnusedMolForBitVectOp_(mols))
        if timing:
            self._state['env_time'] += perf_counter() - t0

        # [3]: BitVectTime
        t0: float = perf_counter() if timing else 0
        for row, mols in zip(p_row, p_bitvect):
            self._EnvData[row, :] = StringVectToNumpy(self._GetBitVectAPI_(mols))
        if timing:
            self._state['fp_time'] += perf_counter() - t0

        return None

    def _ParallelBitVectOp_(self, rows: List[int], safe: bool, SMol: Mol, timing: bool,
                            cached_result: dict) -> None:
        # [1]: Preparation
        StoreEnv: bool = self._state['storeEnvironment']
        temp_BondIdxList = [self._BondIdxList[row] for row in rows]

        # [2]: EnvTime
        t0: float = perf_counter() if timing else 0
        for i, (mols, bIdx) in enumerate(self._GetEnvsAPI_(SMol, temp_BondIdxList, safe)):
            row: int = rows[i]
            if StoreEnv:
                self._StoreEnvMol_(mols, bondIdx=bIdx, row=row)
            cached_result['row'].append(row)
            cached_result['bv'].append(self._DisableUnusedMolForBitVectOp_(mols))
        if timing:
            self._state['env_time'] += perf_counter() - t0

        return None

    def _LBondInfoOp_(self, rows: List[int], safe: bool, SMol: Mol, timing: bool = False) -> None:
        InvalidStereo: bool = self._state['showInvalidStereochemistry']
        t: float = perf_counter() if timing else 0
        for i, row in enumerate(rows):
            bondIdx: int = self._BondIdxList[row]
            IsNewMol: bool = (i == 0)
            LBI_Vect = self._GetLBondInfoAPI_(SMol, bondIdx, UpdateMol=IsNewMol, safe=safe)
            if IsNewMol and InvalidStereo:
                self._LBICreator.CheckStereochemistry(SMol, LBI_Vect=LBI_Vect)
            self._LBondInfoData[row, 0:len(LBI_Vect)] = LBI_Vect

        if timing:
            self._state['lbi_time'] += perf_counter() - t
        return None

    # --------------------------------------------------------------------------------------------------------------
    # Internal Function
    def RefreshData(self) -> None:
        self._MolTable, self._BondIdxList = None, None
        self._InfoData, self._EnvData, self._LBondInfoData, self._Environment = None, None, None, None

        for key in ('create_bde', 'convert_bde'):
            self._state[key]: int = 0
        for key in ('create_time', 'convert_time'):
            self._state[key]: float = 0.0
        for key in ('env_time', 'fp_time', 'lbi_time',):
            self._state[key]: float = 0.0

        RunGarbageCollection(1)

    def _CacheDataset_(self) -> None:
        self._InfoData = self.GetDataInBlock(request='Info')
        self._EnvData = self.GetDataInBlock(request='EnvData')
        self._LBondInfoData = self.GetDataInBlock(request='LBIData')
        self._Environment = self.GetDataInBlock(request='Env')

    def Activate(self, BitVect: Optional[Tuple[bool, ...]] = None, LBondInfo: bool = True) -> None:
        TestStateByWarning(condition=self.Dataset().testable)
        NumsInput: int = InputState.GetNumsInput()

        # [1]: BitVect
        if BitVect is None:
            BitVect = (True,) * NumsInput

        if BitVect is not None:
            InputFullCheck(BitVect, name='BitVect', dtype='Tuple')
            TestState(len(BitVect) == NumsInput, f'BitVect must have {NumsInput} values.')
            self._state['BitVect'] = tuple(BitVect)
            self._state['BV_Reduce']: bool = any(self._state['BitVect'])

        # [2]: LBondInfo
        InputFullCheck(LBondInfo, name='LBondInfo', dtype='bool')
        self._state['LBondInfo'] = LBondInfo

    def _GetLabels_(self, GetLBILabels: bool = True) -> List[str]:
        labels = self._BVManager.GetLabels()
        if FeatureEngineer.BaseFpLabels is None:
            FeatureEngineer.BaseFpLabels = labels.copy()
        if GetLBILabels:
            labels.extend(self._LBICreator.GetLabels())
        return labels

    def EnvExtractor(self) -> EnvExtractor:
        return self._EnvExtractor

    def BitVectManager(self) -> BVManager:
        return self._BVManager

    def LBICreator(self) -> LBICreator:
        return self._LBICreator

    def BitVectSize(self) -> int:
        return len(self.BitVectManager())

    # ------------------------------------------------------------------------------------------------------------
    # Step 02: Do preparation
    def DefineLabels(self, GetLBILabels: bool = True):
        """ This method pre-construct the feature labels and should only be called once. """
        label = self._GetLabels_(GetLBILabels)
        if isinstance(self.Dataset(), FeatureData):
            dataset: FeatureData = self.Dataset()
            dataset.GetDataBlock('Data').SetColumns(np.array(label, dtype=np.object_))
            dataset.EvalEnvLabelInfo()
        return label

    def _PrepareData(self):
        self._PrepareEnv_()  # [1]: Pre-construct the environment
        self._PrepareData_()  # [2]: Generate Feature Array & Association Cache Value
        self._CacheDataset_()

    def _PrepareEnv_(self):
        """ This method pre-construct the array containing the SMILES of environment. """
        # [1]: Pre-construct the environment
        if not self._state['storeEnvironment']:
            return None
        envLabels = [f'{name}: Environment' for name in InputState.GetNames()] + ['New Index']
        env_datablock = self.Dataset().GetDataBlock('Env')
        env_datablock.SetColumns(envLabels)
        env_datablock.InitData(shape=(self._InfoData.shape[0], len(envLabels)), dtype=FIXED_INFO_NPDTYPE,
                               environment=self.GetKey())

    def _PrepareData_(self):
        # [2]: Generate Feature Array & Association Cache Value
        # [2a]: Set bond/radical environment
        EnvLabel: ndarray = np.array(self._GetLabels_(GetLBILabels=False), dtype=np.object_)
        self.Dataset().GetDataBlock('EnvData').SetColumns(EnvLabel)

        shape: Tuple[int, int] = (self._InfoData.shape[0], EnvLabel.shape[0])
        self.SetDataInBlock(np.zeros(shape=shape, dtype=FIXED_INPUT_NPDTYPE), request='EnvData')

        # [2b]: Set localized bond information
        LBILabel: ndarray = np.array(self._LBICreator.GetLabels(), dtype=np.object_)
        self.Dataset().GetDataBlock('LBIData').SetColumns(LBILabel)

        shape: Tuple[int, int] = (self._InfoData.shape[0], LBILabel.shape[0])
        self.SetDataInBlock(np.zeros(shape=shape, dtype=FIXED_INPUT_NPDTYPE), request='LBIData')

        return None

    def ReportSpeed(self) -> None:
        def ToUnit(accumulate: float, n: int, scale: Union[int, float] = 1e3) -> float:
            return scale * accumulate / n

        if self._state['create_bde'] != 0:
            timer: float = ToUnit(self._state['create_time'], self._state['create_bde'])
            print(f"Executing Time for Creation BDE: {self._state['create_time']:.4f} (s) "
                  f"-~-> Per Unit: {timer:.4f} (ms / bond).")

        full_time: float = self._state['create_time']
        if self._state['convert_bde'] != 0:
            timer: float = ToUnit(self._state['convert_time'], self._state['convert_bde'], 1e6)
            print(f"Executing Time for Conversion BDE: {1e3 * self._state['convert_time']:.4f} (ms) "
                  f"-~-> Per Unit: {timer:.4f} (μs / bond)")
            full_time += self._state['create_time']
            timer: float = ToUnit(full_time, self._state['create_bde'] + self._state['convert_bde'])
            print(f"Executing Time for Data Generation in Total: {full_time:.4f} (s) "
                  f"-~-> Per Unit: {timer:.4f} (ms / bond).")

        if not self._state['_TIMER']:
            return None

        n_mol: int = len(self._MolTable) - 1
        n_samples: int = self._state['create_bde']
        print('-' * 80)
        print(f'There are {n_mol} molecules with {n_samples} BDEs that are asked to construct feature from algorithm.')

        total: float = self._state['env_time'] + self._state['fp_time'] + self._state['lbi_time']

        def CastTime(t: float) -> Tuple[float, ...]:
            return t, ToUnit(t, n_samples), ToUnit(t, n_mol)

        time_map: Dict[str, Tuple[float, ...]] = \
            {
                'TOTAL': CastTime(total), 'ENV': CastTime(self._state['env_time']),
                'FP': CastTime(self._state['fp_time']), 'LBI': CastTime(self._state['lbi_time']),
                'Other': CastTime(self._state['create_time'] - total)
            }
        for key, value in time_map.items():
            x, y, z = value
            print(f'Time Spent ({key}): {x:.4f} (s) -~-> {y:.4f} (ms / bond) or {z:.4f} (ms / mol).')

        TIME_RATIO = [self._state['env_time'], self._state['fp_time'], self._state['lbi_time']]
        MIN_TIME_RATIO = min(TIME_RATIO)
        for i, val in enumerate(TIME_RATIO):
            if val == MIN_TIME_RATIO:
                TIME_RATIO[i] = 1
            else:
                TIME_RATIO[i] = float(f'{val / MIN_TIME_RATIO:.2f}')
        print(f'RATIO: (1) ENV: {TIME_RATIO[0]}; (2) FP: {TIME_RATIO[1]}; (3) LBI: {TIME_RATIO[2]} ')
        print('-' * 80)
        return None

    @staticmethod
    def ExportConfiguration(FilePath: str, StorageFolder: str) -> None:
        from Bit2Edge.utils.file_io import FixPath
        if StorageFolder is None:
            SaveDataConfig(FilePath)
        return SaveDataConfig(FixPath(StorageFolder, extension='/') + FilePath)
