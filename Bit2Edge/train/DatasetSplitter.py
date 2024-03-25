# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
#  This .py is doing the data splitter for the object of class:Dataset
# --------------------------------------------------------------------------------

from logging import warning, info
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split

from Bit2Edge.config.splitConfig import GetTrainingState, IsHeldOutValidation, ValidateTrainingKey
from Bit2Edge.config.splitConfig import SPLIT_FRAMEWORK as tFramework
from Bit2Edge.dataObject.DataBlock import GetDtypeOfData
from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.dataObject.FileParseParams import FileParseParams
from Bit2Edge.train.SplitParams import SplitParams
from Bit2Edge.utils.cleaning import RunGarbageCollection
from Bit2Edge.utils.helper import GetIndexOnArrangedData, FindRepeatLine, OptIntDtype
from Bit2Edge.utils.verify import InputCheckRange, InputFastCheck, InputFullCheck, TestState, TestStateByWarning


class DatasetSplitter:
    """ This class attempted to do data preparation before it is shipped. """

    def __init__(self, data: FeatureData):
        # [1]: Predefined Value - Must-have (Quick check)
        TestState(data.trainable, 'The input :arg:`data` must be trainable.')
        TestState(data.GetDataBlock('EnvData').GetData(environment='Test') is not None,
                  msg='The testing data is not null.')
        TestState(data.GetDataBlock('EnvData').GetData(environment='Val') is None,
                  msg='The validation data is not null.')
        self._data: FeatureData = data

        # [2]: Computed Value
        self._DataParams: FileParseParams = self._data.GetParams()
        self._state: Dict[str, Any] = {'MolData': None, 'BondData': None, 'ratio': None}
        self._PreCompute_()

    def _PreCompute_(self) -> None:
        data = self._data.GetDataBlock('Info').GetData(environment='Train')
        mol, bondIndex = self._DataParams.Mol(), self._DataParams.BondIndex()
        self._state['MolData'] = GetIndexOnArrangedData(data, cols=mol, get_last=False, keys=str)
        self._state['BondData'] = GetIndexOnArrangedData(data, cols=[mol, bondIndex], get_last=False, keys=(str, int))

    # [1]: Split Section for SCRATCH_TRAIN -----------------------------------------------------------------------
    @staticmethod
    def _Verify_(ratio: Union[int, float, Tuple], params: SplitParams) -> None:
        """ This method verify to ensure the datatype when splitting the dataset into three
            different set (train, validation, test). """
        ValidateTrainingKey(params.SplitKey)
        TestState(params.SplitKey in tFramework, 'The splitting key should be verified first (should not be called).')
        InputFullCheck(ratio, name='ratio', dtype='int-float-Tuple', delimiter='-')
        if InputFastCheck(ratio, dtype='Tuple'):
            TestState(len(ratio) == 3, f'arg::splitRatio={ratio} should have three inputs.')

        InputFullCheck(params.GoldenRuleForDivision, name='goldenRule', dtype='bool')

    @staticmethod
    def _ComputeSplitRatioNumeric_(ratio: Union[int, float], InputSize: int, params: SplitParams) -> List[float]:
        TrainingStatus: Tuple[bool, ...] = GetTrainingState(params.SplitKey)
        if isinstance(ratio, int):
            InputCheckRange(ratio, name='splitRatio', maxValue=InputSize, minValue=0, rightBound=True)
            devSize: float = ratio / InputSize
        else:
            InputCheckRange(ratio, name='splitRatio', maxValue=1, minValue=0, allowFloatInput=True,
                            leftBound=False, rightBound=False)
            devSize: float = int(InputSize ** ratio) / InputSize if params.GoldenRuleForDivision else ratio
        ratio: List[float] = [0, 0, 0]
        if TrainingStatus[1]:
            ratio[1] = devSize
        if TrainingStatus[2]:
            ratio[2] = devSize
        ratio[0] = 1 - ratio[1] - ratio[2]
        return ratio

    @staticmethod
    def _ComputeSplitRatioDefined_(ratio: Tuple, InputSize: int, params: SplitParams) -> List[float]:
        from Bit2Edge.utils.verify import InputCheckHomogenous

        TrainingStatus: Tuple[bool, ...] = GetTrainingState(params.SplitKey)
        TRAIN, _, _ = ratio
        InputCheckHomogenous(*ratio, name='ratio', dtype='int-float', delimiter='-')

        for index in range(1, 3):
            InputCheckRange(ratio[index], name=f'ratio[{index}]', maxValue=InputSize, minValue=0,
                            allowFloatInput=isinstance(TRAIN, float), leftBound=True,
                            rightBound=isinstance(TRAIN, int))

        ratio = list(ratio)
        total: Union[int, float] = sum(ratio)
        TestState(total > 0, 'ratio cannot be calculated.')

        if (isinstance(TRAIN, int) and total != InputSize) or (not isinstance(TRAIN, int) and total != 1):
            warning('The ratio would be rescaled.')
            ratio[0] /= total
            ratio[1] /= total
            ratio[2] /= total
            print(f'Updating Ratio: {ratio}.')

        # Worked when validation set is not an independent object or not available
        if 0 <= params.SplitKey < 10:
            if params.TransferRatio is None:
                params.TransferRatio = 0.5

            InputCheckRange(params.TransferRatio, name='TransferRatio', maxValue=1, minValue=0,
                            allowNoneInput=False, allowFloatInput=True, leftBound=True, rightBound=True)

            if TrainingStatus[1] and not TrainingStatus[2]:
                if ratio[2] != 0:
                    info(f'TransferRatio={params.TransferRatio} due to the specific behavior of this train scheme.')
                    ratio[0] += params.TransferRatio * ratio[2]
                    ratio[1] += (1 - params.TransferRatio) * ratio[2]
                    ratio[2] = 0
            elif TrainingStatus[2] and not TrainingStatus[1]:
                if ratio[1] != 0:
                    info(f'TransferRatio={params.TransferRatio} due to the specific behavior of this train scheme.')
                    ratio[0] += params.TransferRatio * ratio[1]
                    ratio[2] += (1 - params.TransferRatio) * ratio[1]
                    ratio[1] = 0
            else:
                raise ValueError('Error Source Code.')
        return ratio

    def _ComputeRatio_(self, ratio: Union[int, float, Tuple], params: SplitParams) -> Tuple[float, ...]:
        """
            This method will calculate the user-defined ratio that matched with our rule.
            If ratio is set to be int | float, TransferRatio argument will not be applied.
        """
        if params.SplitKey == -1:
            state = (1.0, 0.0, 0.0)
            self.SetState('ratio', state)
            return state

        TEMP = {'sample': 'samples', 'mol': 'molecules', 'bond': 'bonds', }
        TestState(params.mode in TEMP.keys(), f'The algorithm only support these values: {[*TEMP]}.')
        if params.mode == 'sample':
            InputSize: int = self._data.GetDataBlock('Info').GetData(environment='Train').shape[0]
        elif params.mode == 'mol':
            InputSize: int = len(self.GetState('MolData'))
        else:
            InputSize: int = len(self.GetState('BondData'))
        print(f'There are {InputSize} {TEMP[params.mode]} waiting.')

        if isinstance(ratio, (int, float)):
            func: Callable = DatasetSplitter._ComputeSplitRatioNumeric_
        else:
            func: Callable = DatasetSplitter._ComputeSplitRatioDefined_

        ratio = func(ratio=ratio, InputSize=InputSize, params=params)
        self.SetState(keyword='ratio', value=tuple(ratio))
        return self.GetState('ratio')

    def _Distribute_(self, row: Union[List[int], ndarray, slice, Tuple[int, int]],
                     source_feature_set: str, target_feature_set: str) -> None:
        """
        This method will copy a part of data from this set (:arg:`fromSet`) to another set (:arg:`toSet`).
        The original data in this set (:arg:`fromSet`) is not lost. If you passed a slice,
        the data is only a view.
        """

        if isinstance(row, Tuple):
            row = list(range(*row))
        for _, attr in enumerate(self._data.GetDataBlockName()):
            source_data = self._data.GetDataBlock(attr).GetData(environment=source_feature_set)
            if source_data is not None:
                self._data.GetDataBlock(attr).SetData(source_data[row, :], environment=target_feature_set)
        RunGarbageCollection(0)

    @staticmethod
    def _Split_(inputs: Union[ndarray, List[Tuple[int, str]]], TestSize: float, seed: int,
                *args, **kwargs) -> Tuple[Union[ndarray, List[Tuple[int, str]]], ...]:
        return train_test_split(inputs, test_size=TestSize, random_state=seed, *args, **kwargs)

    @staticmethod
    def _TrySortMask_(*inputs: List[Union[int, Tuple[int, str]]], params: SplitParams):
        if not params.sorting:
            return None

        def KeySortFunction(value):
            return value if params.mode == 'sample' else value[int(params.objectSorting)]

        for arr in inputs:
            arr.sort(key=KeySortFunction, reverse=params.reverse)
        return None

    def _2FactorSplit_(self, seed: int, params: SplitParams) -> None:
        # [1]: Setup Configuration
        Train, Dev, Test = GetTrainingState(params.SplitKey)
        TestState((0 <= params.SplitKey <= 9) and Train,
                  f'The pipeline {self.Split} is falsely assigned task or has trouble.')

        # [2]: Split data
        factor = self.GetState('ratio')[(1 if Dev else 2)]
        InputSize: int = self._data.GetDataBlock('Info').GetData(environment='Train').shape[0]
        if params.mode == 'sample':
            datatype = OptIntDtype((1, InputSize))
            TrainLine, TestLine = DatasetSplitter._Split_(np.arange(start=0, stop=InputSize, step=1, dtype=datatype),
                                                          TestSize=factor, seed=seed)
            DatasetSplitter._TrySortMask_(TrainLine, TestLine, params=params)

        else:
            CheckUpData: List[Tuple] = self.GetState('MolData') if params.mode == 'mol' else self.GetState('BondData')
            _, TestMolLine = DatasetSplitter._Split_(CheckUpData, TestSize=factor, seed=seed)

            DatasetSplitter._TrySortMask_(TestMolLine, params=params)

            TrainLine, TestLine = \
                self._ExtractSamples_(TestMolLine, params=params, GetRemainder=True)

        # [3]: Distribute data: Advanced index made a copy so we don't want double copy
        SetMsg = GetDtypeOfData('feature_set')
        if Test:
            self._Distribute_(row=TestLine, source_feature_set=SetMsg[0], target_feature_set=SetMsg[2])
            self._Distribute_(row=TrainLine, source_feature_set=SetMsg[0], target_feature_set=SetMsg[0])
        else:
            self._Distribute_(row=TestLine, source_feature_set=SetMsg[0], target_feature_set=SetMsg[1])
            if IsHeldOutValidation(params.SplitKey):
                self._Distribute_(row=TrainLine, source_feature_set=SetMsg[0], target_feature_set=SetMsg[0])
        return None

    def _3FactorSplit_(self, seed: int, params: SplitParams) -> None:
        # [1]: Setup Configuration & Prepare Data
        configuration = GetTrainingState(params.SplitKey)
        TestState(params.SplitKey >= 10 and all(configuration),
                  f'The pipeline {self.Split} is falsely assigned task or has trouble.')
        InputSize: int = self._data.GetDataBlock('Info').GetData(environment='Train').shape[0]
        mask: ndarray = np.ones(shape=(InputSize,), dtype=np.bool_)

        # [1]: Choosing coefficient
        RATIO: Tuple[float, float, float] = self.GetState('ratio')
        if params.TrainDevSplit:
            factor: float = RATIO[2]
            factor_2: float = RATIO[1] / (RATIO[1] + RATIO[0])
        else:
            factor: float = RATIO[1] + RATIO[2]
            factor_2: float = RATIO[1] / (RATIO[1] + RATIO[2])

        # [2]: Split Data
        if params.mode == 'sample':
            DTYPE = OptIntDtype((1, InputSize))
            TrainLine, TestLine = DatasetSplitter._Split_(np.arange(start=0, stop=InputSize, step=1, dtype=DTYPE),
                                                          TestSize=factor, seed=seed)

            if params.TrainDevSplit:
                if params.SplitKey in (10, 15):
                    TrainLine, DevLine = self._Split_(TrainLine, TestSize=factor_2, seed=seed)
                else:
                    size: int = TrainLine.size
                    if params.SplitKey in (11, 16):
                        separator: int = int(size * (1 - factor_2))
                        DevLine = TrainLine[slice(separator, size, 1)]
                    else:
                        separator: int = int(size * factor_2)
                        DevLine = TrainLine[slice(0, separator, 1)]

            else:
                if params.SplitKey in (10, 15):
                    TestLine, DevLine = DatasetSplitter._Split_(TestLine, TestSize=factor_2, seed=seed)
                else:
                    size: int = TestLine.size
                    if params.SplitKey in (11, 16):
                        separator: int = int(size * (1 - factor_2))
                        DevLine = TestLine[slice(separator, size, 1)]
                        TestLine = TestLine[slice(0, separator, 1)]
                    else:
                        separator: int = int(size * factor_2)
                        DevLine = TestLine[slice(0, separator, 1)]
                        TestLine = TestLine[slice(separator, size, 1)]

            DatasetSplitter._TrySortMask_(TrainLine, TestLine, DevLine, params=params)

        else:
            CheckUpData: List[Tuple] = self.GetState('MolData').copy() if params.mode == 'mol' else \
                self.GetState('BondData').copy()

            # [2]: Split data
            if params.TrainDevSplit:
                TrainMolLine, TestMolLine = DatasetSplitter._Split_(CheckUpData, TestSize=factor, seed=seed)
                if params.SplitKey in (10, 15):
                    _, DevMolLine = DatasetSplitter._Split_(TrainMolLine, TestSize=factor_2, seed=seed)
                else:
                    size: int = len(TrainMolLine)
                    if params.SplitKey in (11, 16):
                        separator: int = int(size * (1 - RATIO[1] / RATIO[0]))
                        DevMolLine = TrainMolLine[slice(separator, size, 1)]
                    else:
                        separator: int = int(size * RATIO[1] / RATIO[0])
                        DevMolLine = TrainMolLine[slice(0, separator, 1)]

            else:
                _, TestMolLine = DatasetSplitter._Split_(CheckUpData, TestSize=factor, seed=seed)
                if params.SplitKey in (10, 15):
                    TestMolLine, DevMolLine = DatasetSplitter._Split_(TestMolLine, TestSize=factor_2, seed=seed)
                else:
                    size: int = len(TestMolLine)
                    if params.SplitKey in (11, 16):
                        separator: int = int(size * (1 - RATIO[1] / RATIO[0]))
                        DevMolLine = TestMolLine[slice(separator, size, 1)]
                        TestMolLine = TestMolLine[slice(0, separator, 1)]
                    else:
                        separator: int = int(size * RATIO[1] / RATIO[0])
                        DevMolLine = TestMolLine[slice(0, separator, 1)]
                        TestMolLine = TestMolLine[slice(separator, size, 1)]

            TestLine = self._ExtractSamples_(TestMolLine, params=params, GetRemainder=False)
            DevLine = self._ExtractSamples_(DevMolLine, params=params, GetRemainder=False)

            DatasetSplitter._TrySortMask_(TestLine, DevLine, params=params)

        mask[TestLine] = 0
        if IsHeldOutValidation(params.SplitKey):
            mask[DevLine] = 0

        SetMsg = GetDtypeOfData('feature_set')
        self._Distribute_(TestLine, source_feature_set=SetMsg[0], target_feature_set=SetMsg[2])
        self._Distribute_(DevLine, source_feature_set=SetMsg[0], target_feature_set=SetMsg[1])
        self._Distribute_(np.nonzero(mask)[0], source_feature_set=SetMsg[0], target_feature_set=SetMsg[0])

    def _ExtractSamples_(self, rows: List[Tuple[int, Union[int, str]]], params: SplitParams,
                         GetRemainder: bool = True) -> Union[Tuple[List[int], ...], List[int]]:
        """
        This method is extract all the rows that for dataset splitting.

        Arguments:
        ---------

        rows : List[Tuple[int, Union[int, str]]]
            This is a list of tuple containing the group of object we want to extract. It is the result 
            of :meth:`Utility.GetIndexOnArrangedData()`.

        params : SplitParams
            The parameter used for data splitting
        
        GetRemainder : bool, optional
            If True, we return the remainder of the rows that are not extracted. Otherwise, we return
            the extracted rows only. (default is True).

        Returns:
        -------
                - If :arg:`GetRemainder`=True, return two lists of integer.
                - If :arg:`GetRemainder1=False, return one list of integer.
        """

        # [0]: Preparing data
        InfoData: ndarray = self._data.GetDataBlock('Info').GetData(environment='Train')
        maxSize: int = InfoData.shape[0]

        # [1]: Find the next row that is compatible with the condition
        ExtractedSamples: List[int] = []
        boolMask: List[bool] = [False] * maxSize

        # Stop condition (below) -> Rewrite
        # If IsMolOp=True (rows=MolData), condition: InfoData[next_row, self._mol] != InfoData[row, self._mol]
        # If IsMolOp=False (rows=BondData), condition: InfoData[next_row, self._mol] != InfoData[row, self._mol] or
        # InfoData[next_row, self._bondIdx] != InfoData[row, self._bondIdx]
        col: int = self._DataParams.Mol() if params.mode == 'mol' else self._DataParams.BondIndex()
        for row, value in rows:
            ExtractedSamples.append(row)
            boolMask[row] = True
            tSmiles: str = InfoData[row, self._DataParams.Mol()]
            for next_row in range(row + 1, maxSize):
                if InfoData[next_row, self._DataParams.Mol()] != tSmiles or InfoData[next_row, col] != value:
                    break
                ExtractedSamples.append(next_row)
                boolMask[next_row] = True

        if params.ZeroDuplicate:  # Remove duplication in radicals
            RemoveLine: List[int] = \
                FindRepeatLine(array=InfoData[ExtractedSamples, :], cols=self._DataParams.Radical(),
                               keyCol=self._DataParams.Mol(), aggCols=None, removeReverseState=params.StrictCleaning)
            if len(RemoveLine) != 0:
                for line in RemoveLine:
                    boolMask[ExtractedSamples[line]] = False
                ExtractedSamples: object = np.delete(ExtractedSamples, obj=RemoveLine, axis=None).tolist()

        if not GetRemainder:
            return ExtractedSamples

        return [i for i in range(maxSize) if not boolMask[i]], ExtractedSamples

    def Split(self, ratio: Union[int, float, Tuple], seed: int, params: SplitParams) -> None:
        """
        This method is used in the train phase or retraining phase, which split our feature set into
        one to three different set which is controlled according to the :arg:`params.SplitKey`.

        Arguments:
        ---------

        ratio : int, float, or Tuple
            If a tuple, we defined a fixed ratio for our three feature set. If integer, we defined the number
            of samples each for validation set and test set. If a float, we defined a particular ratio
            for validation set and test set (:arg:`params.GoldenRuleForDivision`=False).
                - request (str): This is the feature we want to apply, it is usually the
                    train set ('Train').

        seed : int
            The seed for pseudo-random splitting algorithm (used in :meth:`sklearn.train_test_split()`)

        params : SplitParams
            A collection of split parameters used for dataset division

        """
        # [1]: Pre-compute + Data Verification
        print('Splitting Dataset - Ratio: ', end='')
        DatasetSplitter._Verify_(ratio=ratio, params=params)
        self._ComputeRatio_(ratio=ratio, params=params)
        TRAIN, DEV, TEST = self.GetState('ratio')
        print(f'Train: {TRAIN:.4f} - Val: {DEV:.4f} - Test: {TEST:.4f} >--> Val/Train: {(DEV / TRAIN):.4f}.')

        if params.SplitKey == -1:
            return None

        TestStateByWarning(IsHeldOutValidation(params.SplitKey), 'Validation set is trained along with Training set.')

        if 0 <= params.SplitKey <= 9:
            TestState(GetTrainingState(params.SplitKey)[0], msg='Wrong assigned task.')
            self._2FactorSplit_(seed=seed, params=params)
        else:
            TestState(params.SplitKey >= 10 and all(GetTrainingState(params.SplitKey)), msg='Wrong assigned task.')
            self._3FactorSplit_(seed=seed, params=params)
        RunGarbageCollection(0)

    def SliceDataset(self, params: SplitParams, DevSize: Optional[Tuple[int, int]] = None,
                     TestSize: Optional[Tuple[int, int]] = None, TrainOnValidation: bool = False, ) -> None:
        """ This method is used to slice the train set into other set."""
        if DevSize is None and TestSize is None:
            warning('No implementation can be further done. Disable implementation.')
            params.SplitKey = -1
            return None

        if True:
            InputFullCheck(DevSize, name='DevSize', dtype='Tuple-None', delimiter='-')
            InputFullCheck(TestSize, name='TestSize', dtype='Tuple-None', delimiter='-')
            InputFullCheck(TrainOnValidation, name='TrainOnValidation', dtype='bool')

            TestState(len(DevSize) == 2, 'DevSize should contain only two positive numerical values.')
            TestState(len(TestSize) == 2, 'TestSize should contain only two positive numerical values.')

            SampleSize: int = self._data.GetDataBlock('Info').GetData(environment='Train').shape[0]
            InputCheckRange(DevSize[0], name='DevSize[0]', maxValue=DevSize[1], minValue=0)
            InputCheckRange(DevSize[1], name='DevSize[1]', maxValue=SampleSize, minValue=0)
            InputCheckRange(TestSize[0], name='TestSize[0]', maxValue=TestSize[1], minValue=0)
            InputCheckRange(TestSize[1], name='TestSize[1]', maxValue=SampleSize, minValue=0)

        SetMsg = GetDtypeOfData('feature_set')
        mask: List[int] = [False] * SampleSize
        if TestSize is not None:
            for value in range(TestSize[0], TestSize[1]):
                mask[value] = True
            self._Distribute_(row=slice(*TestSize), source_feature_set=SetMsg[0], target_feature_set=SetMsg[2])

        if DevSize is not None:
            if not TrainOnValidation:
                for value in range(DevSize[0], DevSize[1]):
                    mask[value] = True
            self._Distribute_(row=slice(*DevSize), source_feature_set=SetMsg[0], target_feature_set=SetMsg[1])

        index: List[int] = [i for i in range(0, SampleSize) if not mask[i]]
        if sum(index) == len(index) * (index[0] + index[-1]) // 2:
            index: slice = slice(index[0], index[-1])
        self._Distribute_(row=index, source_feature_set=SetMsg[0], target_feature_set=SetMsg[0])

        if TestSize is not None and DevSize is not None:
            params.SplitKey = 10 if not TrainOnValidation else 15
        elif TestSize is not None and DevSize is None:
            params.SplitKey = 0
        else:
            params.SplitKey = 1 if not TrainOnValidation else 2
        return None

    # ----------------------------------------------------------------------------
    def _ValidateKeyword_(self, keyword: str, strict: bool = True) -> None:
        if strict:
            TestState(keyword in self._state, f'The keyword::{keyword} was not found in the storage.')
        return None

    def GetState(self, keyword: str, strict: bool = True) -> Any:
        self._ValidateKeyword_(keyword=keyword, strict=strict)
        return self._state[keyword]

    def SetState(self, keyword: str, value: Any, strict: bool = True) -> None:
        self._ValidateKeyword_(keyword=keyword, strict=strict)
        self._state[keyword] = value
