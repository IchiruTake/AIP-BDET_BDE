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
from typing import Optional, Union, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray

from Bit2Edge.config.startupConfig import GetPrebuiltInfoLabels
from Bit2Edge.dataObject.DataBlock import DEFAULT_OUTPUT_NPDTYPE
from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.dataObject.DatasetLinker import DatasetLinker
from Bit2Edge.dataObject.FileParseParams import FileParseParams
from Bit2Edge.dataObject.InputState import InputState
from Bit2Edge.input.Creator import FeatureEngineer
from Bit2Edge.input.BondParams import BondParams
from Bit2Edge.input.MolProcessor.MolEngine import MolEngine
from Bit2Edge.molUtils.molUtils import CanonMolString
from Bit2Edge.test.TesterUtilsP1 import ConfigureEngineForTester, EnableGPUDevice
from Bit2Edge.test.TesterUtilsP2 import TestSafeTester
from Bit2Edge.test.params.PredictParams import PredictParams
from Bit2Edge.utils.cleaning import DeleteArray, RunGarbageCollection
from Bit2Edge.utils.helper import ReadFile, Sort, KeySortParams
from Bit2Edge.utils.verify import InputFullCheck, TestState

BASE_TEMPLATE = Optional[ndarray]
INPUT_FOR_DATABASE = Union[BASE_TEMPLATE, pd.DataFrame]
ITERABLE_TEMPLATE = Union[INPUT_FOR_DATABASE, List, Tuple]
SINGLE_INPUT_TEMPLATE = Union[BASE_TEMPLATE, str]
MULTI_COLUMNS = Union[List, Tuple]


def _Fix2DData_(database: INPUT_FOR_DATABASE) -> ndarray:
    if not isinstance(database, ndarray):
        database: ndarray = np.asarray(database)
    if database.ndim < 2:
        database: ndarray = np.atleast_2d(database)
    TestState(database.ndim == 2, 'We only allow 2D-array.')
    return database


def _TuneFinalDataset_(InfoData: ndarray, FalseLine: List[int], TrueReference: Optional[ndarray] = None) \
        -> Tuple[ndarray, Optional[ndarray]]:
    InputFullCheck(FalseLine, name='FalseLine', dtype='List')
    if len(FalseLine) != 0:
        FalseLine.sort()
        info(f'These are the error lines needed to be removed: {FalseLine}')
        InfoData: ndarray = DeleteArray(InfoData, obj=FalseLine, axis=0)
        if TrueReference is not None:
            TrueReference: ndarray = DeleteArray(TrueReference, obj=FalseLine, axis=0)
    return InfoData, TrueReference


class BaseTester(DatasetLinker):
    """
        This class is a controlled template class to prepare the prediction of the Bond Dissociation
        Energy by the Bit2Edge architecture and AIP-BDET model.
        This class does not have the attribute of _TF_Model or related can be found here.
    """

    def __init__(self, dataset: FeatureData, generator: FeatureEngineer,
                 GPU_MEM: bool = False, GPU_MASK: Tuple[bool, ...] = (True,)):
        if GPU_MEM:
            EnableGPUDevice(mask=GPU_MASK)

        # [1.0]: Main Attribute for Data
        dataset, generator = ConfigureEngineForTester(dataset, generator)
        super(BaseTester, self).__init__(dataset=dataset)
        self._generator: FeatureEngineer = generator
        self.Dataset().GetDataBlock('Info').SetColumns(GetPrebuiltInfoLabels())
        self._params = FileParseParams(mol=0, radical=(1, 2), bIdx=3, bType=4)  # Cache attribute

        # [2]: Status Attribute
        self._timer: Dict[str, Union[int, float]] = \
            {
                'add': 0.0, 'create': 0.0, 'process': 0.0, 'predictMethod': 0.0,
                'predictFunction': 0.0, 'visualize': 0.0
            }

    # [0]: Indirect/Hidden Method: -----------------------------------------------------------------------------------
    # [0.1]: Preprocessing - Timer: ----------------------------------------------------------------------------------
    def ResetTime(self) -> None:
        for key, _ in self._timer.items():
            self._timer[key] = 0.0

    def GetFullTime(self) -> float:
        return self._timer['add'] + self.GetProcessTime() + self._timer['visualize']

    def GetProcessTime(self) -> float:
        return self._timer['create'] + self._timer['process'] + self._timer['predictMethod']

    def GetTimeReport(self) -> pd.DataFrame:
        """ Return a DataFrame that recorded execution time """
        index = ['#1: Adding Data', '#2: Create Data', '#3: Process Data', '#4: Prediction Method',
                 '#5: Prediction Speed', '#6: Visualization Speed', '#7: Total Time']
        column = ['Full Timing (secs)', 'Timing per Unit (ms/bond)']

        def ToUnit(timing: float, samples: int, scale: Union[int, float] = 1e3) -> float:
            return scale * (timing / samples)

        N: int = self.Dataset().GetDataBlock('Info').GetData(environment='Test').shape[0]
        value = [[self._timer['add'], ToUnit(self._timer['add'], N)],
                 [self._timer['create'], ToUnit(self._timer['create'], N)],
                 [self._timer['process'], ToUnit(self._timer['process'], N)],
                 [self._timer['predictMethod'], ToUnit(self._timer['predictMethod'], N)],
                 [self._timer['predictFunction'], ToUnit(self._timer['predictFunction'], N)],
                 [self._timer['visualize'], ToUnit(self._timer['visualize'], N)],
                 [self.GetFullTime(), ToUnit(self.GetFullTime(), N)]]

        x = pd.DataFrame(data=value, index=index, columns=column, dtype=DEFAULT_OUTPUT_NPDTYPE)
        x.index.name = 'Method'
        return x

    # [1]: Adding Data: ----------------------------------------------------------------------------------------------
    # [1.0]: Adding Data: --------------------------------------------------------------------------------------------
    def _TestInfoData_(self):
        InfoBlock = self.GetDataInBlock(request='Info')
        MSG: str = 'The data is not ready for prediction. Please check the data state'
        TestState(InfoBlock is not None, msg=f'{MSG} at InfoData.')
        if self.GetTargetReference() is not None:
            TestState(InfoBlock.shape[0] == self.GetTargetReference().shape[0],
                      'The number of observations in the data and target are not compatible.')
        return None

    def _TestFeatureData_(self):
        self._TestInfoData_()
        MSG: str = 'The data is not ready for prediction. Please check the data state.'
        TestState(self.GetDataInBlock('LBIData') is not None, msg=f'{MSG} at LBIData.')
        TestState(self.GetDataInBlock('EnvData') is not None, msg=f'{MSG} at EnvData.')

    def Generator(self) -> FeatureEngineer:
        return self._generator

    def GetParams(self) -> FileParseParams:
        return self._params

    def RefreshData(self):
        self.Dataset().RefreshData()
        self.Dataset().GetDataBlock('Info').SetColumns(GetPrebuiltInfoLabels())
        self.Generator().RefreshData()
        self.ResetTime()

    def StartNewProcess(self) -> float:
        self.RefreshData()
        return perf_counter()

    def _ReadCsvFile_(self, path: str, params: FileParseParams, sorting: bool = False,
                      ascending: bool = True) -> ndarray:
        InputFullCheck(sorting, name='sorting', dtype='bool')
        InputFullCheck(ascending, name='ascending', dtype='List-bool-Tuple', delimiter='-')
        useCols: List[int] = params.DistributeData(useTarget=True)
        if not sorting:
            return ReadFile(FilePath=path, header=0, usecols=useCols, get_values=True, get_columns=False)

        DataFrame: pd.DataFrame = ReadFile(FilePath=path, header=0, usecols=useCols, get_values=False,
                                           get_columns=False)
        if params.BondIndex() is not None:
            modified_ascending: List[bool] = [ascending, ascending] if isinstance(ascending, bool) else ascending
            by = [DataFrame.columns[self._params.Mol()], DataFrame.columns[self._params.BondIndex()]]
            DataFrame = DataFrame.sort_values(by=by, ascending=modified_ascending,
                                              inplace=False)
        else:
            DataFrame = DataFrame.sort_values(by=DataFrame.columns[self._params.Mol()],
                                              ascending=ascending, inplace=False)
        return DataFrame.values

    # [3.1]: Single-Adding Method: ---------------------------------------------------------------------------------
    def _UpdateTiming_(self, start_time: float, dtype: str, timer_type: Optional[str]) -> None:
        TestState(dtype in self._timer, 'There are no equivalent timing method found.')
        self._timer[dtype] += perf_counter() - start_time
        if timer_type is None:
            return None
        print(f'{timer_type} Time: {self._timer[dtype]:.4f} (s).')

    def AddListOfMol(self, mols: Union[List[str], Tuple[str, ...], object], mode: str = 'SMILES',
                     canonicalize: bool = True, useChiral: bool = True,
                     params: Optional[BondParams] = None) -> List[List]:
        """
        Reading a list of molecule by either Smiles or InChi Key. Note that if a string-like
        molecule is incorrectly represented, it will be discarded out of the consideration.

        Arguments:
        ---------

        mols : List[str], Tuple[str, ...], or object
            A sequence of molecule by string datatype
        
        mode : str
            The representation type of molecule in current. Default to 'SMILES'.
            Extra options are 'InChi' and 'SMARTS'.
        
        canonicalize : bool
            If True, the string-molecule would be standardized by the universal scheme
            defined by RDKit. Default to True.
        
        useChiral : bool
            If True, we fixed the current chirality has been defined in the string-molecule
            (isomericSmiles). Default to True. Note that the chirality is only be defined
            by SMILES notation.
                
        params : BondParams
            The parameters for the method `MolEngine().GetBonds()`. Default to None.

        Returns:
        -------

        A list of valid molecules

        """
        InputFullCheck(mols, name='mols', dtype='List-Tuple', delimiter='-')
        InputFullCheck(mode, name='mode', dtype='str')
        InputFullCheck(canonicalize, name='canonicalize', dtype='bool')
        InputFullCheck(useChiral, name='useChiral', dtype='bool')

        # [1]: Initialization & Verification/Testing
        invalid_attempt: List = []
        valid_attempt: List[str] = []
        timer: float = self.StartNewProcess()
        for idx, user_mol in enumerate(mols):
            try:
                # Error molecule raised here
                mol = CanonMolString(user_mol, mode=mode, useChiral=useChiral)
                valid_attempt.append(mol)
            except (RuntimeError, ValueError, TypeError):
                invalid_attempt.append((idx, user_mol))

        # [2]: Display the error molecule
        if len(invalid_attempt) != 0:
            print(f'There are {len(invalid_attempt)} molecules that would be deleted.')
            print(pd.DataFrame(data=invalid_attempt, index=None, columns=['No.', 'User_Mol']), '\n')

        self._UpdateTiming_(start_time=timer, dtype='add', timer_type='Adding')

        return self._CreateInfo(smiles=valid_attempt, params=params)

    def AddMol(self, mol: str, mode: str = 'SMILES', canonicalize: bool = True, useChiral: bool = True,
               params: Optional[BondParams] = None) -> List[List]:
        InputFullCheck(mol, name='mol', dtype='str')
        return self.AddListOfMol(mols=[mol], mode=mode, canonicalize=canonicalize, useChiral=useChiral,
                                 params=params)

    AddMol.__doc__ = AddListOfMol.__doc__

    def AddMolFile(self, FilePath: str, mode: str = 'SMILES', mol: int = 0, sorting: bool = False,
                   ascending: bool = True, canonicalize: bool = True, useChiral: bool = True,
                   params: Optional[BondParams] = None) -> List[List]:
        FileParams = FileParseParams(mol=mol, radical=None, bIdx=None, bType=None, target=None, strict=False)
        database: ndarray = self._ReadCsvFile_(FilePath, params=FileParams, sorting=sorting, ascending=ascending)
        return self.AddListOfMol(mols=database.tolist(), mode=mode, canonicalize=canonicalize, useChiral=useChiral,
                                 params=params)

    AddMolFile.__doc__ = AddListOfMol.__doc__

    def _CreateInfo(self, smiles: List[str], params: Optional[BondParams] = None) -> List[List]:
        """
        This method called directly the MolEngine().GetBonds(mol for mol in valid_mols, *args, **kwargs)
        Check the documentation of the method above for more information.

        Arguments:
        ---------

        smiles : List[str]
            A sequence of molecule by string datatype. This parameter is a result of `BaseTester.AddListOfMol()`.
        
        params : BondParams
            The parameters for the method `MolEngine().GetBonds()`. Default to None.
        
        Returns:
        -------
        
        A list of information of the molecule.

        """
        timer: float = perf_counter()
        reactions: List[List] = MolEngine().GetBondOnMols(smiles=smiles, params=params)
        v_reactions = [reaction[2:] for reaction in reactions]
        self.SetDataInBlock(np.array(v_reactions, dtype=np.object_), request='Info')
        self._UpdateTiming_(start_time=timer, dtype='add', timer_type='Adding')
        return v_reactions

    def AddOnDefinedArray(self, database: ndarray, params: FileParseParams, isZeroIndex: bool = True,
                          sorting: bool = False, ascending: bool = True) -> None:
        """
        This method will read molecule from a well-constructed array (completed information).

        Arguments:
        ---------

        database : ndarray
            An numpy array of information
        
        params : FileParseParams
            The parameters to determine the structure of the :arg:`database`.
        
        isZeroIndex : bool
            Whether to guarantee if all the index is starting from zero.
            If False, all the bond index will be decrease by one. Default to True.
        
        sorting : bool
            Whether to implement sorting. Sorting may reduce feature engineering
            time by a significant factor. Default to False.
        
        ascending : bool
            Whether to sort the database in ascending order. Default to True.
        
        """
        # Hyper-parameter Verification
        database: ndarray = _Fix2DData_(database)
        InputFullCheck(isZeroIndex, name='isZeroIndex', dtype='bool')
        InputFullCheck(sorting, name='sorting', dtype='bool')

        timer: float = self.StartNewProcess()
        if not isZeroIndex:
            database[:, params.BondIndex()] = np.array(database[:, params.BondIndex()], dtype=np.uint16) - 1
        if sorting:
            key = KeySortParams(status=sorting, column=params.Mol(), key=str, reverse=not ascending,
                                maxSize=database.shape[1])
            task = KeySortParams(status=True, column=params.BondIndex(), key=int, reverse=False,
                                 maxSize=database.shape[1])
            database = Sort(database, KeyParams=key, SortColParams=task)

        self.SetDataInBlock(np.array(database[:, params.DistributeData(useTarget=False)], dtype=np.object_),
                            request='Info')
        if params.Target() is not None:
            self.SetDataInBlock(np.array(database[:, params.Target()], dtype=DEFAULT_OUTPUT_NPDTYPE), request='Target')
        return self._UpdateTiming_(start_time=timer, dtype='add', timer_type=None)

    def AddOnDefinedFile(self, FilePath: str, params: FileParseParams, isZeroIndex: bool = True,
                         sorting: bool = False, ascending: bool = True) -> None:
        database: ndarray = self._ReadCsvFile_(path=FilePath, params=params, sorting=sorting, ascending=ascending)
        self.AddOnDefinedArray(database=database, params=params, isZeroIndex=isZeroIndex, sorting=False,
                               ascending=ascending)
        return None

    AddOnDefinedFile.__doc__ = AddOnDefinedArray.__doc__
    """ You can write some code that allow this class to exploit the predictor to construct data with 
        atomic index, bond index, or even the radical-like molecule using some searching. """

    # [4]: Prediction Data: ------------------------------------------------------------------------------------------
    @TestSafeTester
    def FeatureEngineering(self, BitVectState: Optional[Tuple[bool, ...]] = None, LBondInfoState: bool = True) -> None:
        """
        This function will request the generator to create features. Note that this function did not
        run the data final processing. This should be implemented in the subclass.

        Arguments:
        ---------
        
        UseMolCount : bool
            If True, the timing verbose is measured by the number of molecules. If False, the timing verbose is
            measured by the number of reactions. Default to False.
        
        BitVectState : Tuple[bool, ...]
            This argument control whether the bit-vect is generated or not. Default to `(True, ) *
            InputState.GetNumsInputByRadiusLayer()`
        
        LBondInfoState : bool
            Whether to generate the local bond information. Default to True.
        
        """
        # Hyper-parameter Verification
        self._TestInfoData_()
        if self.GetTargetReference() is not None:
            TestState(self.GetDataInBlock(request='Info').shape[0] == self.GetTargetReference().shape[0],
                      'The number of observations in the data and target are not compatible.')
        self._generator.RefreshData()

        # [1]: Generate Features
        print('-' * 80)
        print('Feature Generator: RUNNING ... Please wait for a sec ...')
        timer: float = perf_counter()
        self._generator.Activate(BitVect=BitVectState, LBondInfo=LBondInfoState)
        self._generator.GetDataAPI(GC=True, SAFE=False)
        self._UpdateTiming_(start_time=timer, dtype='create', timer_type='Feature-Creation')
        RunGarbageCollection(0)

    def CreateData(self) -> None:
        raise NotImplementedError('This method should be implemented on the subclass.')

    CreateData.__doc__ = FeatureEngineering.__doc__

    def predict(self, params: PredictParams) -> pd.DataFrame:
        raise NotImplementedError('This method should be implemented on the subclass.')

    def ExportInfoToDataframe(self):
        pass