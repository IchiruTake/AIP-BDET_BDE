# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module is to perform the Data Cleaning
# --------------------------------------------------------------------------------
import gc
from logging import warning, info
from time import perf_counter
from typing import Callable, List, Optional, Tuple, Union, Set, Dict

import numpy as np
import pandas as pd
from numpy import ndarray

from Bit2Edge.utils.verify import TestState, TestStateByWarning, InputFastCheck, InputFullCheck, InputCheckRange


# --------------------------------------------------------------------------------

def RunGarbageCollection(generation: int = None, verbose: bool = False) -> int:
    """
    This function is a simple replacement of :meth:`gc.collect()`, but are attached
    with small modifications ahead for GC-debugging.
    """
    InputCheckRange(generation, name='generation', minValue=0, maxValue=2, leftBound=True, rightBound=True,
                    allowNoneInput=True, allowFloatInput=False)
    InputFullCheck(verbose, name='verbose', dtype='bool')
    if generation is None:
        generation: int = 2
    NumLeftoverObjects: int = gc.collect(generation=generation)
    if verbose is True:
        print(f'The number of unreachable objects are {NumLeftoverObjects}.')
    return NumLeftoverObjects


def GetRemainingIndexToLimit(PrunedMask: Union[ndarray, List[int], Tuple[int, ...]], maxValue: int) -> List[int]:
    """
    This function will compute the remaining index from 0 to :arg:`maxValue` - 1, but not being
    duplicated in the :arg:`PrunedMask`.

    Arguments:
    ---------

    PrunedMask : ndarray
        The considered array for masking

    maxValue : int
        The maximum threshold value

    Returns:
    -------

    A list of unsigned integer
    """
    if InputFastCheck(PrunedMask, dtype='ndarray'):
        TestState(PrunedMask.ndim == 1, 'indexArray should be 1-dimensional array.')
        PrunedMask = PrunedMask.tolist()
    else:
        TestState(len(set(PrunedMask)) == len(PrunedMask),
                  'PrunedMask should not contained duplicated value.')

    if isinstance(maxValue, (int, np.integer)):
        TestState(maxValue > 0, f'maxValue={maxValue} should be positive integer.')
        TestStateByWarning(maxValue >= max(PrunedMask),
                           f'maxValue={maxValue} should be the largest value of all.')
    else:
        raise TypeError(f'maxValue={maxValue} should be positive integer.')

    TestState(all(int(value) >= 0 for value in PrunedMask), 'PrunedMask should contain only positive integer (>=0).')
    hashMap: Set[int] = set(PrunedMask)
    return [idx for idx in range(0, maxValue) if idx not in hashMap]


# --------------------------------------------------------------------------------
__ITERABLE_0 = Union[ndarray, List]
__ITERABLE_0_5 = Union[ndarray, List, Tuple]
__ITERABLE_1 = Union[ndarray, List, Tuple, pd.Index]

_CLEANING: Dict[str, Union[int, bool]] = \
    {
        'DensityThreshold': 0,  # Used at FeatureCleaning
        'BinaryMode': True,  # Constant, please not modified. Used at FeatureCleaning.
        'np.delete': False,
    }
_FAST_PATH_NUM: int = 512


def ArrayEqual(array_1: __ITERABLE_1, array_2: __ITERABLE_1, allowCache: bool = True) -> bool:
    """
    This function served to be an alias-replacement of :meth:`np.array_equal()` which has hugh amount of overhead.
    The function :meth:`np.array_equal()` always result in O(K*N), depended on task dependency, and K is ranging from
    2 to 4. In the general scenario, where all input arguments are passed validation, it has to ensure that all
    value in array should be converted into boolean matrix and validate using `bool(np.asarray(a==b).all())`.

    However, we want to reduce them the time complexity in the specific task only. Costing O(k) real-time complexity
    only with no extra space complexity O(1) compared to :meth:`np.array_equal()`.

    """
    InputFullCheck(array_1, name='array_1', dtype='List-Tuple-ndarray', delimiter='-')
    InputFullCheck(array_2, name='array_2', dtype='List-Tuple-ndarray', delimiter='-')
    if InputFastCheck(array_1, dtype='List-Tuple', delimiter='-') and \
            InputFastCheck(array_2, dtype='List-Tuple', delimiter='-'):
        # Fast-path for list-tuple on feature-matrix
        if len(array_1) != len(array_2):
            return False
        return not any(v1 != v2 for v1, v2 in zip(array_1, array_2))

    TestState(array_1.ndim == array_2.ndim and array_1.ndim == 1,
              msg=f'Two array was not equivalent in size a: {array_1.shape} --- b: {array_2.shape}.')
    if array_1.size != array_2.size:
        return False

    InputFullCheck(allowCache, name='allowCache', dtype='bool')
    if allowCache:
        return ArrayEqual(array_1.tolist(), array_2.tolist(), False)
    return np.array_equal(array_1, array_2)


def GetLocationForLabelRemoval(SourceLabels: __ITERABLE_0_5, TargetLabels: __ITERABLE_0_5) -> List[int]:
    """
    This method is attempted to identify the positions inside the :arg:`SourceLabels` so that
    when removing all value of those position, you can get the :arg:`TargetLabels`.

    Arguments:
    ---------

    SourceLabels : List[str], or Tuple[str, ...], or ndarray
        The set of labels for label removal.

    TargetLabels : List[str], or Tuple[str, ...], or ndarray
        The set of labels for to achieve by pruning column in :arg:`SourceLabels`.

    Returns:
    -------
        - A list of integer specifying the column for label removal on :arg:`SourceLabels`.
    """
    # [1]: Hyper-parameter Verification
    InputFullCheck(SourceLabels, name='SourceLabels', dtype='List-Tuple-ndarray', delimiter='-')
    InputFullCheck(TargetLabels, name='TargetLabels', dtype='List-Tuple-ndarray', delimiter='-')
    TestState(len(TargetLabels) <= len(SourceLabels),
              msg=f'Two arrays above cannot be matched as the :arg:`TargetLabels` ({len(TargetLabels)}) '
                  f'is longer than the :arg:`SourceLabels` ({len(SourceLabels)}).')

    def StringCastOp(value) -> str:
        return value if isinstance(value, str) else str(value)

    HashedSourceLabels = set(SourceLabels if not isinstance(SourceLabels, ndarray) else SourceLabels.tolist())
    TestState(all(StringCastOp(value) in HashedSourceLabels for value in TargetLabels),
              msg='In the :arg:`TargetLabels` found at least one value is not in :arg:`SourceLabels`.')

    FalseLabel: List[int] = []
    size: int = len(TargetLabels)
    j: int = 0
    for i, label in enumerate(SourceLabels):
        if j < size and StringCastOp(label) == StringCastOp(TargetLabels[j]):
            j += 1
        else:
            FalseLabel.append(i)

    # Validation condition is determined as one of two options below
    # 1) len(SourceLabels) - len(FalseLabel) == len(TargetLabels)
    # 2) j == size <- Unable to reach the target till the end
    TestStateByWarning(j == size or len(SourceLabels) - len(FalseLabel) == len(TargetLabels),
                       msg=f'Two arrays above cannot be matched as the comparison is not completed: '
                           f'Demonstrated from {len(SourceLabels)} to {len(TargetLabels)} by {len(FalseLabel)}.')
    return FalseLabel


# [4.2]: Cleaning Function
def _TestCleaning_(array: ndarray, labels: __ITERABLE_0, co_arrays: Tuple[ndarray, ...]) -> None:
    """ Evaluate whether these arrays can be matchable. """
    InputFullCheck(array, name='array', dtype='ndarray')
    InputFullCheck(labels, name='labels', dtype='ndarray-List', delimiter='-')

    n: int = array.shape[1]
    prefix: str = 'Invalid Length of'
    TestState(len(labels) == n, f'{prefix} Columns ({n} vs {len(labels)}).')
    for i, co_array in enumerate(co_arrays):
        InputFullCheck(co_array, name=f'co_array_{i + 1}', dtype='ndarray')
        TestState(co_array.shape[1] == n, f'{prefix} co_array_{i + 1} ({n}-{co_array.shape[1]}).')
    return None


def _ToSparseMatrix_(array: ndarray, BinaryMode: bool, boolMask: List[bool]) -> Tuple[List[List[int]], List[ndarray]]:
    """
        Convert the array to the custom sparse. Only the column evaluated to be False in
        :arg:`boolMask` is estimating its sparse state.
    """
    from Bit2Edge.utils.helper import OptIntDtype
    TestState(len(boolMask) == array.shape[1], 'Source Code Error.')
    print('Generate Dynamic Sparse Matrix: PENDING ... ')

    n: int = array.shape[1]
    BinaryCols: List[List[int]] = [np.where(array[:, col] == 1)[0].tolist() if not boolMask[col] else []
                                   for col in range(0, n)]
    ExtraCols: List[Union[ndarray, List[int]]] = [[] for _ in range(0, n)]

    if BinaryMode:
        return BinaryCols, ExtraCols

    empty: ndarray = np.array([], dtype=array.dtype)
    opt: Callable = OptIntDtype
    maximumValue: int = np.iinfo(array.dtype).max
    for col in range(0, n):
        if not boolMask[col] and array[:, col].sum() != len(BinaryCols[col]):
            index_temp: ndarray = np.where(np.logical_and(array[:, col] != 1, array[:, col] != 0))[0]
            if index_temp.size == 0:
                ExtraCols[col] = empty
            else:
                ExtraCols[col] = np.zeros(shape=(index_temp.size, 2),
                                          dtype=opt((int(maximumValue), int(index_temp[-1]))))
                ExtraCols[col][:, 0] = index_temp
                ExtraCols[col][:, 1] = array[index_temp, col]

    print('Generate Dynamic Sparse Matrix: DONE ... ')
    return BinaryCols, ExtraCols


def DeleteArray(arr: Union[ndarray, List[str]], obj: List[int],
                axis: Optional[int] = 0) -> Union[ndarray, List[str]]:
    if _CLEANING['np.delete'] is True:
        return np.delete(arr, obj=obj, axis=axis)

    SIZE: int = arr.shape[axis] if axis is not None else (len(arr) if not isinstance(arr, ndarray) else arr.size)
    UPDATED_MASK: List[int] = GetRemainingIndexToLimit(PrunedMask=obj, maxValue=SIZE)
    if axis == 0:
        result = arr[UPDATED_MASK, :]
    elif axis == 1:
        result = arr[:, UPDATED_MASK]
    elif axis is None:
        if isinstance(arr, ndarray):
            result = arr[UPDATED_MASK]
        else:
            result = [arr[idx] for idx in UPDATED_MASK]
    else:
        raise ValueError('Not supported scenario.')
    if InputFastCheck(arr, dtype='ndarray'):
        TestState(result.base is None or not np.may_share_memory(arr, result),
                  msg='Error Numpy Library, Fancy Indexing should create a copy.')
    return result


def _ComputeMaskAtAxe1_(array: ndarray, boolMask: List[bool]) -> None:
    """
    This function is to compute the array by axis = 1. If the feature density is zero, or below the threshold,
    the :arg:`boolMask` would be evaluated to True to remove a feature.
    """

    def reduce(arr: ndarray, mode: str, axis: Optional[int] = 0) -> Union[ndarray, int, object, List[int]]:
        if mode == 'sum':
            return np.sum(arr, axis=axis, dtype=np.uint32).tolist()
        elif mode == 'count':
            return np.count_nonzero(arr, axis=axis).tolist()
        raise ValueError

    # boolMask is initialized with False value as the default but this algorithm can by-pass some checking
    LENGTH: int = array.shape[0]
    SELECTED_SIZE: int = min(_FAST_PATH_NUM * 2, LENGTH)
    IS_INPUT_SMALLER_FASTPATH = (SELECTED_SIZE == LENGTH)
    THRESHOLD: float = _CLEANING['DensityThreshold'] * LENGTH

    ACCEPT_NONZERO_DENSITY: bool = (_CLEANING['DensityThreshold'] != 0)
    BINARY: bool = _CLEANING['BinaryMode']
    info(f'You are using the first-and-last {SELECTED_SIZE // 2} inputs to fast-search the algorithm.'
         f'This is equivalent as {(100 * SELECTED_SIZE / LENGTH):2.3f} (%).')

    # This condition would result in same value regardless of input feature.
    # Just a speed-up on NumPy as `np.count_nonzero` use `np.sum` with `boolean` mask in Python
    if IS_INPUT_SMALLER_FASTPATH:
        FastArr = array
    else:
        FastArr = np.concatenate((array[0:(SELECTED_SIZE // 2)], array[-(SELECTED_SIZE // 2):]), axis=0)
    METHOD: str = 'sum' if BINARY else 'count'
    FastReduce = reduce(FastArr, mode=METHOD, axis=0)

    for column in range(0, array.shape[1]):
        if boolMask[column] is True:  # Extra cost is here, but to reduce code vulnerable
            continue
        density: int = FastReduce[column]
        if not ACCEPT_NONZERO_DENSITY and density not in (0, SELECTED_SIZE):
            # Found equal or more than two distinct values (100 % confidence)
            # Fast-path is only enable if DensityThreshold = 0
            boolMask[column] = False
        else:
            # This feature can have one distinct value or more or equal than two distinct values
            if not IS_INPUT_SMALLER_FASTPATH or ACCEPT_NONZERO_DENSITY:
                t_reduce: int = reduce(array[:, column], mode=METHOD, axis=None)
                if not IS_INPUT_SMALLER_FASTPATH and t_reduce in (0, LENGTH):  # Fully-unique
                    boolMask[column] = True
                elif ACCEPT_NONZERO_DENSITY:
                    if t_reduce <= THRESHOLD or (BINARY and t_reduce >= LENGTH - THRESHOLD):
                        boolMask[column] = True
                else:
                    boolMask[column] = False
            else:
                boolMask[column] = False

    return None


def DeleteDataByMask(mask: List[int], array: ndarray, labels: Union[ndarray, List[str]], *co_arrays: ndarray) -> Tuple:
    array = DeleteArray(array, obj=mask, axis=1)
    labels = DeleteArray(labels, obj=mask, axis=None)
    RunGarbageCollection(0)
    pointer = None
    if co_arrays:
        pointer = [None] * len(co_arrays)
        for i, co_array in enumerate(co_arrays):
            pointer[i] = DeleteArray(co_array, obj=mask, axis=1) if co_array is not None else None
            RunGarbageCollection(0)
    return array, pointer, labels


def ComputeMaskForFeatureCleaning(array: ndarray, labels: Union[ndarray, List[str]], *co_arrays: ndarray) -> List[int]:
    """
    This function is to perform Data Cleaning by Features : When called, it would observe all the
    features in the array (ndarray) from 0 to `ndarray.size - nonTouchableSize`. If any features
    contained constant-singleton value throughout (i.e all 0s / 1s), that features is marked as
    useless and would be removed.

    Arguments:
    ---------

    array : ndarray
        The considered array or identifying useless features.

    labels : List or ndarray
        The labels of :arg:`array` argument.

    nonTouchableSize : int
        The number of features on the right of :arg:`array`, which is not used for cleaning

    co_array : ndarray
        The considered array for feature-adaptation. The size must be the same as :arg:`array`
        (array.shape[1]) and :arg:`labels` (len(labels)).

    Returns:
    -------

    A list of column index showing the column (marked as True) can be cleaned.
    """
    # [1]: Evaluation
    TestStateByWarning(_CLEANING['BinaryMode'] is True,
                       'The mode _CLEANING[BinaryMode] should be True to efficiently run.')
    TestStateByWarning(_CLEANING['DensityThreshold'] == 0,
                       'The mode _CLEANING[DensityThreshold] is NOT zero, you accept a feature density.')
    _TestCleaning_(array, labels, co_arrays)

    print('-' * 30, 'Mask Computation', '-' * 30)
    start: float = perf_counter()
    BoolMask: List[bool] = [False] * array.shape[1]
    print('Number of Original Features:', array.shape[1])

    _ComputeMaskAtAxe1_(array, boolMask=BoolMask)
    RunGarbageCollection(0)
    temp: int = sum(BoolMask)
    MASK: List[int] = [column for column in range(0, array.shape[1]) if BoolMask[column] is True]
    print('\tNumber of Removing Features:', temp)
    print('\tNumber of Remaining Features:', array.shape[1] - temp)
    print(f'Non-Labeled (Data) Cleaning: {perf_counter() - start:.4f} (s) .')
    return MASK


