# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This class served as a general-purpose, intermediate object to predict data
# given one model placeholder or a group of model's placeholder having the same
# input structure before CleanEnvData() function.
# Two sub-classes are :class:`SingleTester` and :class:`GroupTester`. The difference
# between two classes is that :class:`SingleTester` only supported one placeholder,
# whilst :class:`GroupTester` supported multiple placeholders using GroupPlaceholder
# Supported Functions:
# :class:`SingleTester`: Visualize Last Layer + Molecule Drawing
# :class:`GroupTester`: Molecule Drawing
# --------------------------------------------------------------------------------

from logging import info, warning
from time import perf_counter, sleep
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from rdkit.Chem.rdchem import RWMol

from Bit2Edge.dataObject.DataBlock import FIXED_INPUT_NPDTYPE, DEFAULT_OUTPUT_NPDTYPE
from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.dataObject.InputState import InputState
from Bit2Edge.input.Creator import FeatureEngineer
from Bit2Edge.input.MolProcessor.MolDrawer import MolDrawer
from Bit2Edge.input.MolProcessor.MolEngine import MolEngine
from Bit2Edge.molUtils.molUtils import SmilesToSanitizedMol, CanonMolString
from Bit2Edge.test.BaseTester import BaseTester
from Bit2Edge.test.TesterUtilsP1 import CastTarget
from Bit2Edge.test.TargetDefinition import TargetDefinition
from Bit2Edge.test.params.MolDrawParams import MolDrawParams
from Bit2Edge.test.params.PredictParams import PredictParams
from Bit2Edge.test.placeholder.BasePlaceholder import BasePlaceholder
from Bit2Edge.utils.cleaning import RunGarbageCollection
from Bit2Edge.utils.file_io import FixPath
from Bit2Edge.utils.helper import GetIndexOnArrangedData
from Bit2Edge.utils.verify import (InputCheckRange, InputFullCheck, MeasureExecutionTime,
                                   TestState, TestStateByWarning)
from Bit2Edge.test.placeholder.GroupPlaceholder import GroupPlaceholder
from Bit2Edge.test.placeholder.SinglePlaceholderV1 import SinglePlaceholderV1
from Bit2Edge.test.placeholder.SinglePlaceholderV2 import SinglePlaceholderV2


AcceptedPlaceholder = Union[SinglePlaceholderV1, SinglePlaceholderV2, GroupPlaceholder]

class IntermediateTester(BaseTester):

    def __init__(self, dataset: FeatureData, generator: FeatureEngineer, placeholder: AcceptedPlaceholder,
                 GPU_MEM: bool = False, GPU_MASK: Tuple[bool, ...] = (True,)):
        # The automatic configuration must be loaded before initialization
        super(IntermediateTester, self).__init__(dataset=dataset, generator=generator,
                                                 GPU_MEM=GPU_MEM, GPU_MASK=GPU_MASK)
        self._placeholder: AcceptedPlaceholder = placeholder
        if isinstance(placeholder, (SinglePlaceholderV1, SinglePlaceholderV2)) and placeholder.GetTFModel() is None:
            self._placeholder.SetupTFModel(dataset=self.Dataset())

        self._compute_cache: Dict[str, Optional[Union[ndarray, List[str], bool]]] = \
            {
                'prediction-data': None,        # Stored the model's prediction
                'prediction-label': None,

                'performance-label': None,      # The full output including target reference
                'performance-data': None,

                'analysis-label': None,         # The last's layer of single-model prediction
                'analysis-data': None,
            }
        pass

    # [1]: Feature Engineering: --------------------------------------------------------------------------------------
    def CreateData(self, BitVectState: Optional[Tuple[bool, ...]] = None, LBondInfoState: bool = True) -> None:
        # [1]: Input Validation & Feature Engineering
        super(IntermediateTester, self).FeatureEngineering(BitVectState=BitVectState, LBondInfoState=LBondInfoState)

        # [2]: Pre-processing data
        timer: float = perf_counter()
        if isinstance(self.GetPlaceholder(), BasePlaceholder):
            EnvFilePath = self.GetPlaceholder().GetEnvFilePath()
        else:  # This is the GroupPlaceholder
            placeholder: GroupPlaceholder = self.GetPlaceholder()
            NAMES: List[str] = [*placeholder.GetPlaceholders()]
            TestState(len(NAMES) != 0, 'The given :arg:`placeholder` is empty.')
            EnvFilePath: str = placeholder.GetPlaceholder(name=NAMES[0]).GetEnvFilePath()

        self.Dataset().CleanEnvData(EnvFilePath=EnvFilePath, request=self.Dataset().OptFeatureSetKey())
        self._UpdateTiming_(start_time=timer, dtype='process', timer_type='Feature-Reprocessing')
        RunGarbageCollection(0)

    # [2]: Model's Prediction: ------------------------------------------------------------------------------------
    def _TestPredictRequirement_(self, force: bool = False) -> None:
        self._TestFeatureData_()
        placeholder: AcceptedPlaceholder = self.GetPlaceholder()
        if isinstance(placeholder, GroupPlaceholder):
            TestState(len(placeholder) > 0, msg=f'The given :arg:`placeholder`={placeholder.GetName()} is empty.')
        InputFullCheck(force, name='force', dtype='bool')

    def predict(self, params: PredictParams) -> pd.DataFrame:
        raise NotImplementedError

    def GetNumPredictions(self) -> int:
        ComputeCache = self.GetCache()
        return self.GetPlaceholder().GetNumModels() + int(ComputeCache.get('average', 0)) + \
            int(ComputeCache.get('ensemble', 0))

    def GetNumSingleModels(self) -> int:
        return self.GetPlaceholder().GetNumModels()

    def GetYPred(self) -> Optional[ndarray]:
        return self.GetCache()['prediction-data']

    def _LabelCasting_(self):
        CACHE = self.GetCache()
        TestState(CACHE['prediction-data'] is not None, msg='The prediction ndarray is not available.')
        if CACHE['performance-data'] is not None or CACHE['performance-label'] is not None:
            TestStateByWarning(False, msg='The performance data has already been computed -> Doing recompute.')

        result = TargetDefinition.MapTarget(prediction=CACHE['prediction-data'], target=self.GetTargetReference(),
                                            num_pred=self.GetNumPredictions())
        CACHE['prediction-label'] = result['prediction-label']
        CACHE['performance-label'] = result['performance-label']
        CACHE['performance-data'] = result['performance-data']
        return None

    def ExportPredToDf(self, Sfs: Optional[int]) -> pd.DataFrame:
        # [1]: DataFrame of Information
        KEY = self.GetFixedKey()
        df = self.Dataset().InfoToDataFrame(request=KEY, Info=True, Environment=True, Target=False)

        # [2]: DataFrame of Result -> Report
        def mapping(dataframe: pd.DataFrame, labels: Optional[Union[list, ndarray]],
                    result: Optional[ndarray], name: str, sfs: Optional[int]) -> None:
            if labels is None or result is None:
                return None
            TestState(len(labels) == result.shape[1], msg=f'The label and data (in {name}-*) is NOT equivalent.')
            for _, (label, data) in enumerate(zip(labels, result.T)):
                dataframe[label] = data if sfs is None else data.round(sfs)

        InputCheckRange(Sfs, name='Sfs', maxValue=8, minValue=0, allowNoneInput=True)
        CACHE = self.GetCache()
        mapping(df, CACHE['performance-label'], CACHE['performance-data'], name='performance', sfs=Sfs)
        mapping(df, CACHE['analysis-label'], CACHE['analysis-data'], name='analysis', sfs=Sfs)
        return df

    def _DisplayPredTime_(self) -> None:
        print('-' * 80)
        print('The AIP-BDET has finished prediction. Please give us some credit.')
        size: int = self.GetDataInBlock(request='Info').shape[0]

        def _PrintTime_(timer: float, name: str) -> None:
            timing: float = 1e3 * timer / size
            print(f'{name}: {timer:.4f} (s) --> Speed: {timing:.4f} (ms/bond) or {size / timer:.4f} (bond/s).')

        _PrintTime_(self._timer['add'], name='Adding Dataset')
        _PrintTime_(self._timer['create'], name='Create Features')
        _PrintTime_(self._timer['process'], name='Proceed Dataset')
        _PrintTime_(self._timer['predictFunction'], name='Prediction Time on TF_Model')

        if self.GetPlaceholder().GetNumModels() != 1:
            n: int = self.GetPlaceholder().GetNumModels()
            print(f'Number of Models: {n}')
            _PrintTime_(self._timer['predictFunction'] / n, name='Prediction Time on (Each) TF_Model (Avg)')

        _PrintTime_(self._timer['predictMethod'] - self._timer['predictFunction'], name='Constructing DataFrame')
        _PrintTime_(self.GetProcessTime(), name='Full Process')

    # [5]: Other Methods ----------------------------------------------------------------------------------------------
    # [5.1]: Molecule Visualization ----------------------------------------------------------------------------------
    @MeasureExecutionTime
    def DrawMol(self, FolderPath: str = '', RefMode: int = 1, params: Optional[MolDrawParams] = None) -> pd.DataFrame:
        """
        This method is used to draw molecule with RDKit. The output image will be saved in the given directory.

        Arguments:
        ---------

        FolderPath : str
            The directory of output image file. Default to ''.
        
        RefModes : int or Tuple[int, ...]
            The reference outcome to represent. Default to (1, ). Note that if params.NumWeakestTarget is specified.
            The ordering of output DataFrame is sorted by the order of :arg:`RefModes`. If :arg:`RefModes` is None,
            it would choose all settings from the 'last' prediction made.
        
        params : MolDrawParams
            The parameters of molecule drawing. Default to None.
        
        Returns:
        -------

        pd.DataFrame
        """
        # Hyper-parameter Verification
        if True:
            if params is None:
                params = MolDrawParams()
            else:
                params.evaluate()
            self._TestInfoData_()

            InputFullCheck(FolderPath, name='FolderPath', dtype='str')
            if not (FolderPath == '' or FolderPath is None):
                FolderPath = FixPath(FolderPath, extension='/')

            InputFullCheck(RefMode, name='RefMode', dtype='int')

        TargetSet: Tuple[ndarray, str] = self._PrepMolDrawP2_(ref_mode=RefMode, Sfs=params.Sfs)

        print('-' * 30, self.DrawMol, '-' * 30)
        print('RDKit is trying to draw molecule ...')
        info('You should upgrade your RDKit to minimal version of 2022.03.1 to get better image. '
             'See here: https://greglandrum.github.io/rdkit-blog/technical/2022/03/18/refactoring-moldraw2d.html')
        start: float = perf_counter()

        # [1]: Prepare and filter-out using data
        df, IndexData = self._GetMolDrawData_(TargetSet, params=params)

        # [2]: Visualize result
        self._VisualizeMol_(df, IndexData=IndexData, TargetSet=TargetSet, params=params, FolderPath=FolderPath)
        print(f'Executing Time: {perf_counter() - start:.4f} (s).')
        return df

    def _PrepMolDrawP2_(self, ref_mode: int, Sfs: int) -> Tuple:
        TARGET, TARGET_LABEL, DTYPE = self._GetVisualReference_(ref_mode=ref_mode)
        PerfLabel = self.GetCache()['performance-label']
        if PerfLabel is None:
            PerfLabel = []
        AlyLabel = self.GetCache()['analysis-label']
        if AlyLabel is None:
            AlyLabel = []
        if 0 <= ref_mode < len(PerfLabel) + len(AlyLabel):
            TARGET = TARGET.astype(DEFAULT_OUTPUT_NPDTYPE).round(Sfs)
        else:
            TARGET = TARGET.astype(FIXED_INPUT_NPDTYPE)

        return TARGET, TARGET_LABEL

    def _GetMolDrawData_(self, TargetSet: Tuple[ndarray, str], params: MolDrawParams) -> Tuple[pd.DataFrame, List]:
        CACHE = self.GetCache()
        df: pd.DataFrame = pd.DataFrame(data=self.GetDataInBlock(request='Info'), index=None,
                                        columns=self.Dataset().GetDataBlock('Info').GetColumns())
        Sorting = [self.Dataset().GetDataBlock('Info').GetColumns()[self._params.Mol()]]
        for idx, (label, data) in enumerate(zip(CACHE['prediction-label'], CACHE['prediction-data'].T)):
            df[label] = data
        df[TargetSet[1]] = TargetSet[0].ravel()
        if params.SortOnTarget:
            Sorting.append(TargetSet[1])

        df = df.sort_values(Sorting, inplace=False)

        IndexData = GetIndexOnArrangedData(array=df.values, cols=self._params.Mol(), get_last=True)
        if params.NumWeakestTarget is not None:
            NumWeakestTarget: int = params.NumWeakestTarget
            removeLine: List[int] = []
            for row in range(0, len(IndexData) - 1):
                beginLine: int = IndexData[row][0]
                endLine = IndexData[row + 1][0]
                if beginLine + NumWeakestTarget <= endLine:
                    for val in range(beginLine + NumWeakestTarget, endLine, 1):
                        removeLine.append(val)
            if removeLine:
                df = df.drop(removeLine, axis=0)
            IndexData = GetIndexOnArrangedData(array=df.values, cols=self._params.Mol(), get_last=True)
        return df, IndexData

    def _VisualizeMol_(self, df: pd.DataFrame, IndexData: List, TargetSet: Tuple[ndarray, str],
                       params: MolDrawParams, FolderPath: str) -> None:
        BondIdx: ndarray = df.values[:, self._params.BondIndex()]
        TargetLabel: str = TargetSet[1]  # Use the first label only for visualization
        Target: ndarray = df[TargetLabel].values.round(decimals=params.Sfs)

        def DrawMolToFile(m, bond_note: dict, image: str, p: MolDrawParams, legend: Optional[str] = ''):
            mol: RWMol = MolEngine.EmbedBondNote(m, mapping=bond_note, inplace=False)
            mol.RemoveAllConformers()
            MolDrawer.DrawMol2DCairo(mol, mapping=bond_note, filename=image, ImageSize=p.ImageSize, legend=legend)

        SVGs: list = [None] * len(BondIdx)
        for row in range(0, len(IndexData) - 1):
            # [1]: Prepare molecule
            beginLine: int = IndexData[row][0]
            endLine = IndexData[row + 1][0]
            SMILES: str = str(IndexData[row][1])
            basename = beginLine if params.NameImageByStartRow else CanonMolString(SMILES, useChiral=False)

            # [2]: Draw the molecule to the file
            if not params.DenyMolImageFile:
                CanonMol = SmilesToSanitizedMol(SMILES)
                mapping = {int(BondIdx[idx]): f'{int(BondIdx[idx])}: {CastTarget(Target[idx], Sfs=params.Sfs)}'
                           for idx in range(beginLine, endLine)}
                if not params.BreakDownBondImage:
                    DrawMolToFile(m=CanonMol, mapping=mapping, image=f'{FolderPath}{basename}.png',
                                  p=params, legend='')
                else:
                    for bIdx, target in enumerate(mapping):
                        DrawMolToFile(m=CanonMol, mapping={bIdx: target}, image=f'{FolderPath}{basename}.png',
                                      p=params, legend='')

            # [3]: Draw the molecule to the SVG image
            for idx in range(beginLine, endLine):
                mapping = {int(BondIdx[idx]): f'{int(BondIdx[idx])}: {CastTarget(Target[idx], Sfs=params.Sfs)}'}
                svg_image = MolDrawer.DrawMol2DSVG(SMILES, mapping=mapping, ImageSize=params.ImageSize)
                SVGs[idx] = svg_image

            # [3]: Sleep for a while to complete the printing and underlying thread
            sleep(1e-4)
        df['svg'] = SVGs
        return None

    # [0]: Getter & Setter: -------------------------------------------------------------------------------------
    def GetPlaceholder(self) -> AcceptedPlaceholder:
        return self._placeholder

    def RefreshData(self) -> None:
        super(IntermediateTester, self).RefreshData()
        CACHE = self.GetCache()
        for key, _ in CACHE.items():
            CACHE[key] = None if not isinstance(CACHE[key], bool) else False
        return None

    def GetCache(self) -> Dict:
        return self._compute_cache

    # [1]: Visual Referencing: ----------------------------------------------------------------------------------
    def _GetVisualReferenceLastOption_(self) -> Tuple[ndarray, str, object]:
        warning('No compatible labels or valid :arg:`ref_mode` to be FOUND. Switch to the bond type (-1).')
        return self._GetVisualReference_(ref_mode=-1)

    def _GetVisualReference_(self, ref_mode: Optional[int], preserve_dtype: bool = False) \
            -> Tuple[ndarray, str, object]:
        InputFullCheck(ref_mode, name='ref_mode', dtype='int-None', delimiter='-')

        datatype = np.object_
        CACHE = self.GetCache()

        if ref_mode == -1:
            INDEX: int = self._params.BondType()
            target: ndarray = self.GetDataInBlock(request='Info')[:, INDEX:INDEX + 1]
            target_label: str = self.Dataset().GetDataBlock('Info').GetColumns()[INDEX]
            return target, target_label, datatype

        PerfLabel = CACHE['performance-label']

        if 0 <= ref_mode < len(PerfLabel):
            PerfData = CACHE['performance-data']
            TestState(PerfData is not None, 'No prediction attempt is made.')
            INDEX = ref_mode
            target: ndarray = PerfData[:, INDEX:INDEX + 1]
            TestStateByWarning(target[0, 0] != 0.0 or np.unique(target.ravel(), axis=None).size == 1,
                               'No value has been found in this reference mode.')
            target_label: str = PerfLabel[INDEX]
            return target.astype(datatype) if not preserve_dtype else target, target_label, datatype

        AlyLabel = CACHE['analysis-label']

        if len(PerfLabel) <= ref_mode < len(PerfLabel) + len(AlyLabel):
            if isinstance(self.GetPlaceholder(), (SinglePlaceholderV1, SinglePlaceholderV2)):
                return self._GetVisualReferenceLastOption_()

            AlyData = CACHE['analysis-data']
            TestState(AlyData is not None, 'No sub-model prediction attempt is made.')
            INDEX = ref_mode - len(PerfLabel)
            target: ndarray = AlyData[:, INDEX:INDEX + 1]
            TestStateByWarning(target[0, 0] != 0.0 or np.unique(target.ravel(), axis=None).size != 1,
                               'No value has been found in this reference mode.')
            target_label: str = AlyLabel[INDEX]
            return target.astype(datatype) if not preserve_dtype else target, target_label, datatype

        print('Notation:', InputState.GetNames())
        print('LBondInfo Features:', self.Dataset().GetEnvLabelInfo())

        # A variable in FeatureData._AttributeSets_
        request: str = input('Choose your coloring labels (EnvData/LBIData): ')
        if request not in ('EnvData', 'LBIData'):
            raise ValueError('No compatible labels found.')

        LABEL = self.Dataset().GetDataBlock(request).GetColumns()
        FEATURE = self.GetDataInBlock(request=request)
        if isinstance(LABEL, ndarray):
            LABEL = LABEL.tolist()
        INDEX: str = input(f'Choose your coloring features ranging from [0, {len(LABEL) - 1}] or the label: ')

        try:
            if INDEX.isdigit() and 0 <= int(INDEX) < len(LABEL):
                location: int = int(INDEX)
            else:
                location: int = LABEL.index(INDEX)
            return FEATURE[:, location:location + 1], LABEL[location], FIXED_INPUT_NPDTYPE
        except (IndexError, TypeError, ValueError):
            pass

        return self._GetVisualReferenceLastOption_()

    @staticmethod
    def DescribeReferenceMode() -> None:
        """
        The reference mode here is to describe the properties you want to get for the molecule visualization,
        and for the dimensionality-reduction analysis. It is divided into three parts, which is strongly
        referred to the :arg:`ref_mode`.

        >>> TemplateLabel = TargetDefinition.GetTemplatePerfColumnForOneOutput()
        >>> TemplateLabel
        ['Target', 'Predict', 'RelError', 'AbsError']
        >>> print(len(TemplateLabel))
        4

        Given `M` models in the placeholder, each resulted in an output block (denoted by `m`). On each block,
        the model have `N` predictions from 0 to `N`- 1, denoted by `n`. On each prediction with len(TemplateLabel)
        columns, denoted by `i`. For :arg:`ref_mode`, at block `m` and prediction `n`:
            - If i = 0: Using the current reference (i.e Exp/DFT-BDEs).
            - If i = 1: Using the model prediction..
            - If i = 2: Using the relative error.
            - If i = 3: Using the absolute error.

        Special Case:
            - If :arg:`ref_mode`=-1, it used the bond type of each reaction to display.
            - If :arg:`ref_mode` exceeds the above range, last layer visulation is adopted if the provided by the
              single placeholder.
            - If last layer prediction is not applied, it will request to choose an index for visualization.

        Returns:
        -------
            - None
        """
        print('See the document defined by Visual Studio Code.')
