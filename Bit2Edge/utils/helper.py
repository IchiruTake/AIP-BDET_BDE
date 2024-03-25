# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
from collections import defaultdict
from logging import info
from typing import Callable, List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from numpy import ndarray

from Bit2Edge.utils.cleaning import DeleteArray
from Bit2Edge.utils.file_io import ReadFile, ExportFile
from Bit2Edge.utils.verify import TestState, InputFastCheck, InputFullCheck, InputCheckRange, \
    InputCheckIterable, MeasureExecutionTime


# -------------------------------------------------------------------------------------------------------------------
# [5]: Function used for extension
# [5.1]: Function used for extension
def SortColInArray(array: ndarray, column: int, reverse: bool = False) -> ndarray:
    """
    Sorting a column in a ndarray.

    Arguments:
    ---------

    array : ndarray
        The considered array to be sorted.

    column : int
        The column needed to be sorted.

    reverse :bool
        Whether to reverse the order of sorting (default to False),

    Returns:
    -------

    A copied of sorted ndarray
    """
    if not InputFullCheck(array, name='database', dtype='ndarray', warning_only=True):
        array = np.asarray(array)
    InputCheckRange(column, name='column', maxValue=array.shape[1], minValue=0)
    InputFullCheck(reverse, name='reverse', dtype='bool')
    index: ndarray = np.argsort(array[:, column])
    return array[index[::-1]] if reverse else array[index]


def __CheckForGetIndexFunction__(array: ndarray, column: int, state: bool = False) -> ndarray:
    if not InputFastCheck(array, dtype='ndarray'):
        array = np.asarray(array)
    InputCheckRange(column, name='column', maxValue=array.shape[1], minValue=0)
    InputFullCheck(state, name='state', dtype='bool')
    return array


def GetIndexOnArrangedData(array: ndarray, cols: Union[int, List[int], Tuple[int, ...]], get_last: bool = False,
                           keys: Optional[Union[List[Callable], Callable, Tuple[Callable, ...]]] = str) -> List:
    """
    Get the index of the object based on the `column` of the array.

    Arguments:
    ---------

    array : ndarray
        The considered array to get the group of index.
    
    cols : int or List[int]
        The column(s) needed to be considered.
    
    get_last : bool
        Whether to get the last empty line. This can be used as the stopping condition.
        Default to False.
    
    keys : Callable
        The key(s) function needed for grouping. Default to `str`.
    
    Returns:
    -------

    """

    # [1]: Evaluation
    if InputFastCheck(cols, dtype='int') and InputFastCheck(keys, dtype='Callable-None', delimiter='-'):
        def cast(data, row: int, column: int, key: Callable):
            return key(data[row, column]) if key is not None else data[row, column]

        array = __CheckForGetIndexFunction__(array=array, column=cols, state=get_last)
    else:
        def cast(data, row: int, columns: List[int], key: List[Callable]) -> Tuple:
            return tuple(k(data[row, column]) if k is not None else data[row, column]
                         for column, k in zip(columns, key))

        array = __CheckForGetIndexFunction__(array=array, column=cols[0], state=get_last)
        for col in cols:
            __CheckForGetIndexFunction__(array=array, column=col, state=get_last)

    cache = cast(array, 0, cols, keys)
    MolData: List[Tuple] = [(0, cache)]

    for index, value in enumerate(array[1:, cols], start=1):
        result = cast(array, index, cols, keys)
        if result != cache:
            MolData.append((index, result))
            cache = result
    if get_last:
        MolData.append((array.shape[0], None))
    return MolData


def GetIndexOnArrangedDataByDict(array: ndarray, cols: Union[int, List[int], Tuple[int, ...]], get_last: bool = False,
                                 keys: Union[List[Callable], Callable, Tuple[Callable, ...]] = str) -> Dict:
    data = GetIndexOnArrangedData(array, cols, get_last, keys)
    result = defaultdict(list)
    for row, value in data:
        result[value].append(row)
    return result


GetIndexOnArrangedDataByDict.__doc__ = GetIndexOnArrangedData.__doc__


# [5.2]: Function used for advanced extension
def _CheckAggregateCol_(agg_cols: Optional[Union[int, List[int], Tuple]], name: str, maxValue: int,
                        minValue: int = 0, **kwargs) -> Optional[Union[int, List[int], Tuple]]:
    # **kwargs: Argument need for pipeline inputCheckRange
    if agg_cols is None:
        return agg_cols

    InputFullCheck(agg_cols, name=name, dtype='int-List-Tuple', delimiter='-')
    if InputFastCheck(agg_cols, dtype='List-Tuple'):
        for idx, val in enumerate(agg_cols):
            InputCheckRange(val, name=f'{name}[{idx}]', maxValue=maxValue, minValue=minValue, **kwargs)
        return agg_cols

    InputCheckRange(agg_cols, name=name, maxValue=maxValue, minValue=minValue, **kwargs)
    return [agg_cols]


def WrapperOfFile(func: Callable, FilePath: str, Output: str = None, *args, **kwargs) -> pd.DataFrame:
    arr, labels = ReadFile(FilePath=FilePath, header=0, get_values=True, get_columns=True)
    DataFrame = pd.DataFrame(data=func(arr, *args, **kwargs), index=None, columns=labels)
    if Output is not None:
        ExportFile(DataFrame=DataFrame, FilePath=Output)
    return DataFrame


@MeasureExecutionTime
def ToReverseVect(array: ndarray, cols: Tuple[int, int] = (1, 2), removeIdentical: bool = True,
                  insertLast: bool = True) -> ndarray:
    """
    This function will generate a reverse combination into the array.

    Arguments:
    ---------

    array : ndarray
        The considered array.
    
    cols : Tuple[int, int]
        The two columns needed to be considered.
    
    removeIdentical : bool
        Whether to ignore the same identical combination. Default to True.
    
    insertLast : bool
        Whether to place the new combination at the middle or at the end of `array`.
        Default to True. If False, it will be placed at the middle of the array.
    
    Returns:
    -------
    A new ndarray
    """
    # Hyper-parameter Verification
    if True:
        if not InputFastCheck(array, dtype='ndarray'):
            array = np.asarray(array)
        InputCheckIterable(value=cols, name='cols', maxValue=array.shape[1], minValue=0, maxInputInside=2)
        InputFullCheck(removeIdentical, name='removeIdentical', dtype='bool')
        InputFullCheck(insertLast, name='InsertLast', dtype='bool')

    r1, r2 = cols
    if insertLast is False:
        path = np.arange(0, 2 * array.shape[0], 2, dtype=OptIntDtype((0, 2 * array.shape[0])))
        newFile = np.zeros(shape=(array.shape[0] * 2, array.shape[1]), dtype=array.dtype)
        newFile[path, :] = array
        newFile[path + 1, :] = array

        newFile[path + 1, r1] = array[:, r2]
        newFile[path + 1, r2] = array[:, r1]
        if removeIdentical:
            # Note that we only test on the initial value only, stored in path, not for full file
            return DeleteArray(newFile, obj=[row for row in path if newFile[row, r1] == newFile[row, r2]], axis=0)
    else:
        if removeIdentical:
            path = [row for row in range(array.shape[0]) if array[row, r1] != array[row, r2]]
            if len(path) == array.shape[0]:
                path = slice(0, array.shape[0])
        else:
            path = slice(0, array.shape[0])

        newFile = np.concatenate((array, array[path, :]), axis=0)
        newFile[array.shape[0]:, r1] = array[path, r2]
        newFile[array.shape[0]:, r2] = array[path, r1]

    return newFile


def ToReverseVectByFile(FilePath: str, Output: str = None, cols: Union[List[int], Tuple[int]] = (1, 2),
                        removeIdentical: bool = True, insertLast: bool = True) -> pd.DataFrame:
    return WrapperOfFile(ToReverseVect, FilePath=FilePath, Output=Output, cols=cols,
                         removeIdentical=removeIdentical, insertLast=insertLast)


ToReverseVectByFile.__doc__ = ToReverseVect.__doc__


@MeasureExecutionTime
def FindRepeatLine(array: ndarray, cols: Union[int, Tuple[int, int]] = (1, 2), keyCol: int = 0,
                   aggCols: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
                   removeReverseState: bool = True) -> List[int]:
    """
    This function will generate a list of integer showing the row of duplication data
    we need to remove. Which by then passed to :arg:`RemoveRepeatedRadicals()`.

    Arguments:
    ---------

    array : ndarray
        The considered array.
    
    keyCol : int
        The column used to build group data (GetIndexOnArrangedData).
    
    cols : int or Tuple[int, int]
        The two columns of combinations needed to be considered.
    
    aggCols : int or List[int]
        The columns we performed data averaging. Default to None.
    
    removeReverseState : bool
        If True, it would remove all similar combination in either mode A-B or B-A;
        Otherwise, each combination is independent. Default to True.
    
    Returns:
    -------
    
    A list of integer
    """
    # [0]: Hyper-parameter Verification
    array = __CheckForGetIndexFunction__(array, keyCol, removeReverseState)
    aggCols = _CheckAggregateCol_(agg_cols=aggCols, name='TargetCol', maxValue=array.shape[1], minValue=0)
    if InputFastCheck(cols, dtype='int'):
        InputCheckRange(cols, name='cols', maxValue=array.shape[1], minValue=0)
        radicalsList = array[:, cols].tolist()
        removeReverseState = False
    else:
        InputCheckIterable(cols, name='cols', maxValue=array.shape[1], minValue=0, maxInputInside=2)
        radicalsList = array[:, cols].tolist()

    def evaluate(radicals: List[List[str]], r1: int, r2: int, reverse: bool) -> bool:
        if radicals[r1][0] == radicals[r2][0] and radicals[r1][1] == radicals[r2][1]:
            return True
        elif reverse and (radicals[r1][0] == radicals[r2][1] and radicals[r1][1] == radicals[r2][0]):
            return True
        return False

    GroupData = GetIndexOnArrangedData(array, cols=keyCol, get_last=True)
    RemoveLine: List[int] = []
    for index in range(0, len(GroupData) - 1):  # Extract Molecule
        begin, end = GroupData[index][0], GroupData[index + 1][0]
        mask: List[Optional[bool]] = [False] * (end - begin)
        for i in range(begin, end):  # For every bond
            if mask[i - begin] is True:
                continue
            ReducedLine: List[int] = [i]
            for j in range(i + 1, end):
                if mask[j - begin] is False and evaluate(radicalsList, r1=i, r2=j, reverse=removeReverseState):
                    mask[j - begin] = True
                    RemoveLine.append(j)
                    ReducedLine.append(j)

            if aggCols is not None and len(ReducedLine) > 1:
                for value in aggCols:
                    array[i, value] = array[ReducedLine, value].astype(np.float32).mean()
    return RemoveLine


def DelRepeatLines(array: ndarray, cols: Union[int, Tuple[int, int]] = (1, 2), keyCol: int = 0,
                   aggCols: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
                   removeReverseState: bool = True) -> ndarray:
    RemoveLine = FindRepeatLine(array, cols=cols, keyCol=keyCol, aggCols=aggCols,
                                removeReverseState=removeReverseState)
    return DeleteArray(array, obj=RemoveLine, axis=0) if RemoveLine else array


def DelRepeatLinesByFile(FilePath: str, Output: str = None, keyCol: int = 0, cols: Union[int, Tuple[int, int]] = (1, 2),
                         aggCols: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
                         removeReverseState: bool = True) -> pd.DataFrame:
    return WrapperOfFile(DelRepeatLines, FilePath=FilePath, Output=Output, cols=cols, keyCol=keyCol,
                         aggCols=aggCols, removeReverseState=removeReverseState)


DelRepeatLines.__doc__ = FindRepeatLine.__doc__
DelRepeatLinesByFile.__doc__ = FindRepeatLine.__doc__


@MeasureExecutionTime
def DelRepeatLine1Col(array: ndarray, keyCol: int = 0, col: int = 3,
                      aggCols: Optional[Union[int, List[int], Tuple[int, ...]]] = None) -> ndarray:
    """
    This function will remove repeated value within the group by arranged
    value in the :arg:`col`.

    Arguments:
    ---------

    array : ndarray
        The considered array.
    
    keyCol : int
        The column used to build group data (GetIndexOnArrangedData).
    
    col : int
        The column needed to be considered.
    
    aggCols : int or List[int]
        The columns we performed data averaging. Default to None.
    
    Returns:
    -------

    An ndarray
    """
    return DelRepeatLines(array, cols=col, keyCol=keyCol, aggCols=aggCols, removeReverseState=False)


def RemoveRepeatedIn1ColByFile(FilePath: str, Output: str = None, keyCol: int = 0, col: int = 3,
                               aggCols: Union[int, List[int], Tuple] = None) -> pd.DataFrame:
    return WrapperOfFile(func=DelRepeatLine1Col, FilePath=FilePath, Output=Output,
                         keyCol=keyCol, col=col, aggCols=aggCols)


RemoveRepeatedIn1ColByFile.__doc__ = DelRepeatLine1Col.__doc__


class KeySortParams:
    __slots__ = ('status', 'column', 'key', 'reverse')

    def __init__(self, status: bool, column: int, key: Callable,
                 reverse: bool = False, maxSize: int = None) -> None:
        InputFullCheck(status, name='status', dtype='bool')
        InputFullCheck(key, name='key', dtype='Callable')
        InputCheckRange(column, name='column', maxValue=maxSize, minValue=0, rightBound=False)
        InputFullCheck(reverse, name='reverse', dtype='bool')

        self.status: bool = status
        self.column: int = column
        self.key: Callable = key
        self.reverse: bool = reverse


def Sort(array: ndarray, KeyParams: KeySortParams, SortColParams: KeySortParams,
         *NextColParams: KeySortParams) -> ndarray:
    """
    This function will sort a numpy array on two specified columns with defined order.

    Arguments:
    ---------

    array : ndarray
        The considered array.
    
    KeyParams : KeySortParams
        The column used to build group data (GetIndexOnArrangedData).
    
    SortColParams : KeySortParams
        The first column needed to be sorted.
    
    NextColParams : KeySortParams
        A sequence of column needed to be sorted along with :arg:`SortColParams` CONTINUOUSLY by 
        each task within this argument. Default to None.
    
    Returns:
    -------

    An ndarray
    """

    # [1]: Evaluation
    if not InputFastCheck(array, dtype='ndarray'):
        array = np.asarray(array)

    if KeyParams.status:
        array: ndarray = SortColInArray(array, column=KeyParams.column, reverse=KeyParams.reverse)

    GroupData = GetIndexOnArrangedData(array, cols=KeyParams.column, get_last=True)
    for i in range(0, len(GroupData) - 1):
        begin, end = GroupData[i][0], GroupData[i + 1][0]
        if SortColParams.status and end - begin != 1:
            temp: List = array[begin:end, :].tolist()
            temp.sort(key=lambda item: SortColParams.key(item[SortColParams.column]),
                      reverse=SortColParams.reverse)
            array[begin:end, :] = temp

        if NextColParams is not None and len(NextColParams) > 0:
            temp_status: bool = SortColParams.status
            SortColParams.status = False

            array[begin:end, :] = Sort(array[begin:end, :], SortColParams, NextColParams[0], *NextColParams[1:])
            SortColParams.status = temp_status

    return array


def SortByFile(FilePath: str, KeyParams: KeySortParams, SortColParams: KeySortParams,
               *NextColParams: KeySortParams, Output: str = None) -> pd.DataFrame:
    return WrapperOfFile(func=Sort, FilePath=FilePath, Output=Output, KeyParams=KeyParams,
                         SortColParams=SortColParams, NextColParams=NextColParams)


SortByFile.__doc__ = Sort.__doc__


@MeasureExecutionTime
def ArrangeData(array: ndarray, keyCol: int, nextCol: Optional[int] = None,
                mode: str = 'sort') -> ndarray:
    """
    This function will arrange the array from top to bottom.

    Arguments:
    ---------

    array : ndarray
        The considered array.
    
    keyCol : int
        The first column needed to be arranged.
    
    nextCol : int
        The second column needed to be arranged. Default to None.
    
    mode : str
        If :arg:`mode`='sort', we sort the :arg:`nextCol` of the array. Otherwise, we continue arranging
        in this column. Default to 'sort'.
    
    Returns:
    -------

    An ndarray
    """
    # Hyper-parameter Verification
    if True:
        if not InputFastCheck(array, dtype='ndarray'):
            array = np.asarray(array)

        InputCheckRange(keyCol, name='keyCol', maxValue=array.shape[1], minValue=0)
        InputCheckRange(nextCol, name='nextCol', maxValue=array.shape[1], minValue=0, allowNoneInput=True)

        InputFullCheck(mode, name='mode', dtype='str')
        TestState(mode in ('sort', 'arrange'), 'Unable to perform further implementation.')

    # [1]: Retrieve the structure of the data
    hashtable: Dict[str, List[Tuple[int, int]]] = {}
    GroupData = GetIndexOnArrangedData(array, cols=keyCol, get_last=False)
    FoundDuplication: bool = False
    for idx, (row, value) in enumerate(GroupData):
        try:
            hashtable[value].append((row, idx))
            FoundDuplication = True
        except (IndexError, ValueError):
            hashtable[value] = [(row, idx)]

    # [2]: If found duplicate value, we search and rearranged by the following structure
    if FoundDuplication:
        rowLine: List[int] = [0] * array.shape[1]
        maskLine: List[bool] = [False] * len(GroupData)
        counter: int = 0
        for row, value in GroupData:
            if maskLine[row]:  # If it has been found previously, skip it
                continue
            SimilarRows: List[Tuple[int, int]] = hashtable[value]
            for SimilarRow, Index in SimilarRows:
                maskLine[SimilarRow] = True
                for i in range(SimilarRow, GroupData[Index + 1][0]):
                    rowLine[counter] = i
                    counter += 1

        array: ndarray = array[rowLine, :]

    # [3]: Doing task on the second column.
    if keyCol is None:
        return array

    del hashtable, GroupData
    if mode == 'sort':
        key = KeySortParams(status=False, column=keyCol, key=str, reverse=False, maxSize=array.shape[1])
        task = KeySortParams(status=True, column=nextCol, key=int, reverse=False, maxSize=array.shape[1])
        return Sort(array=array, KeyParams=key, SortColParams=task)

    GroupData = GetIndexOnArrangedData(array, cols=nextCol, get_last=True)
    for i in range(0, len(GroupData) - 1):
        START, END = GroupData[i][0], GroupData[i + 1][0]
        array[START:END, :] = GetIndexOnArrangedData(array[START:END, :], )

    return array


def ArrangeDataByFile(FilePath: str, Output: str = None, keyCol: int = 0,
                      nextCol: Optional[int] = None, mode: str = 'sort') -> pd.DataFrame:
    return WrapperOfFile(func=DelRepeatLine1Col, FilePath=FilePath, Output=Output,
                         keyCol=keyCol, nextCol=nextCol, mode=mode)


ArrangeDataByFile.__doc__ = ArrangeData.__doc__


def ComputeErrorHistogram(data: pd.DataFrame, error_column: Union[str, int], index_column: Union[str, int] = None,
                          interval: float = 0.25, maximum_benchmark: float = 5.0, x_axis: str = 'Error (kcal/mol)',
                          y_axis: str = 'Counting', title: str = 'Error Histogram') -> None:
    """
    This function will compute the histogram of the error.

    Arguments:
    ---------

    data : pd.DataFrame
        The considered data.
    
    error_column : str or int
        The column name or index of the error column.
    
    index_column : str or int
        The column name or index of the index column. Default to None.
    
    interval : float
        The interval of the histogram. Default to 0.25.
    
    maximum_benchmark : float
        The maximum value of the benchmark. Default to 5.0.
    
    x_axis : str
        The x-axis label. Default to 'Error (kcal/mol)'.
    
    y_axis : str
        The y-axis label. Default to 'Counting'.
    
    title : str
        The title of the histogram. Default to 'Error Histogram'.
    """

    # Hyper-parameter Verification
    def inputCheckColumn(column: Union[str, int], name: str, df: pd.DataFrame) -> str:
        InputFullCheck(column, name=name, dtype='str-int', delimiter='-')
        if InputFastCheck(column, dtype='int'):
            InputCheckRange(column, name=name, maxValue=df.values.shape[1], minValue=0)
            return df.columns[column]
        elif InputFastCheck(column, dtype='str'):
            TestState(column in list(df.columns), f'Error column must be inside the DataFrame ({column}).')
        return column

    if True:
        InputFullCheck(data, name='data', dtype='DataFrame')
        inputCheckColumn(column=error_column, name='error_column', df=data)

        if index_column is not None:
            inputCheckColumn(column=index_column, name='index_column', df=data)
            InputFullCheck(error_column, name='error_column', dtype='str-int', delimiter='-')

        InputFullCheck(x_axis, name='x_axis', dtype='str')
        InputFullCheck(y_axis, name='y_axis', dtype='str')
        InputFullCheck(title, name='title', dtype='str')

        InputFullCheck(interval, name='interval', dtype='int-float', delimiter='-')
        if interval <= 0:
            info('Your interval is negative. Change to default (0.25).')
            interval = 0.25

        InputFullCheck(maximum_benchmark, name='maximum_benchmark', dtype='int-float', delimiter='-')
        if maximum_benchmark <= 0:
            info('Your maximum_benchmark is negative. Change to default (5).')
            maximum_benchmark = 5
        if maximum_benchmark < interval:
            raise ValueError('Maximum benchmark must exceed the interval.')

        pass

    import matplotlib.pyplot as plt

    def call(DataFrame, objectType: Optional[str], bound: List[float]):
        plt.clf()
        plt.autoscale(DataFrame=False)
        plt.hist(DataFrame, bins=bound, alpha=0.75, color='red', rwidth=0.85)
        plt.xlabel(str(x_axis))
        plt.ylabel(str(y_axis))
        plt.title(str(title) if object_type is None else str(title) + f"({objectType})")
        plt.xlim(0, bound[- 1])
        plt.show()

    edge = [interval * index for index in range(0, int((maximum_benchmark + interval) // interval))] + \
           [maximum_benchmark + interval]

    if index_column is None:
        dataframe = data[error_column].values
        dataframe[dataframe > maximum_benchmark] = maximum_benchmark + interval / 2
        call(DataFrame=dataframe, objectType=None, bound=edge)
    else:
        modified_data = data[[index_column, error_column]].values
        modified_data = SortColInArray(array=modified_data, column=0, reverse=False)
        index_array = GetIndexOnArrangedData(array=modified_data, cols=0, get_last=True)
        for i in range(0, len(index_array) - 1):
            copy = modified_data[index_array[i]:index_array[i + 1], 1]
            copy[copy > maximum_benchmark] = maximum_benchmark + interval / 2
            object_type = index_array[i][1]
            call(DataFrame=copy, objectType=object_type, bound=edge)
    return None


# -------------------------------------------------------------------------------------------------------------------
# [6]: Function used for further extension: Checking data type
def OptIntDtype(bound: Union[Tuple, List]) -> np.dtype:
    """
    This function will return the optimal data type for the given boundary. The boundary you provided 
    should be accurate represent all integer numerical which is used when generating the size for the
    sparse matrix, or row/line (index) for datapoints sampling.

    Arguments:
    ---------

    bound : Tuple or List
        The shape of the array.
    
    Returns:
    -------

    np.dtype
        The optimal data type.
    """
    minimum, maximum = min(bound), max(bound)
    TestState(InputFastCheck(minimum, dtype='int') and InputFastCheck(maximum, dtype='int'),
              f'Float value has been found. Please check your input ({bound}).')

    if minimum >= 0:
        np_dtype = (np.uint8, np.uint16, np.uint32, np.uint64)
    else:
        np_dtype = (np.int8, np.int16, np.int32, np.int64)

    highest_point = max(abs(minimum), abs(maximum))
    for search_dtype in np_dtype:
        if highest_point <= np.iinfo(search_dtype).max:
            return np.dtype(search_dtype)
    return np.dtype(np_dtype[-2])
