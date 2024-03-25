# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

import logging
from logging import warning
from time import perf_counter
from typing import Callable, List, Optional, Tuple, Union, Set, Dict, Any

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.sparse import coo_matrix, spmatrix, csc_matrix, csr_matrix


def CheckCompatibility():
    from rdkit import __version__ as rdkit_version
    from sklearn import __version__ as sklearn_version
    from tensorflow import __version__ as tf_version
    np_major, np_minor, np_patch = np.__version__.split('.')
    pd_major, pd_minor, pd_patch = pd.__version__.split('.')
    rdkit_major, rdkit_minor, rdkit_patch = rdkit_version.split('.')
    sklearn_major, sklearn_minor, sklearn_patch = sklearn_version.split('.')
    tf_major, tf_minor, tf_patch = tf_version.split('.')

    if not (int(np_major) == 1 and int(np_minor) >= 18):
        raise ImportWarning(f'Numpy version is relatively low ({np.__version__}). Please upgrade into version at '
                            f'least 1.18+. Try version 1.20.3.')

    if not (int(pd_major) == 1 and int(pd_minor) >= 2):
        raise ImportWarning(f'Pandas version is relatively low ({pd.__version__}). Please upgrade into version at '
                            f'least 1.2.x. Try version 1.2.4.')

    if not (int(sklearn_major) == 0 and int(sklearn_minor) >= 23):
        raise ImportWarning(f'Scikit-Learn version is relatively low ({sklearn_version}). Please upgrade into version '
                            f'at least 0.23.x. Try version 0.24.x+.')

    if not ((int(rdkit_major) == 2020 and int(rdkit_minor) == 9) or int(rdkit_major) >= 2021):
        raise ImportError(f'RDKit version is relatively low ({rdkit_version}). Please upgrade into version at '
                          'least 2020.09.x.')

    if not (int(tf_major) == 2 and int(tf_minor) >= 5):
        raise ImportError(f'TensorFlow version is relatively low ({tf_version}). Please upgrade into version 2.5.x.')

    try:
        from dask import __version__ as dask_version
        dask_major, dask_minor, dask_patch = dask_version.split(".")
        if not (int(dask_major) == 2021 and int(dask_minor) == 4):
            raise ImportWarning(f'Dask version is relatively low ({dask_version}). Please upgrade into version at '
                                f'least 2021.04.x. Try version 2021.04.x.')
    except (ImportError, ImportWarning):
        pass


def TestState(condition: bool, msg: Optional[str] = None, error=None) -> None:
    if condition:
        return None
    if isinstance(error, str):
        msg = error
        error = ValueError
    elif error is None:
        error = ValueError
    raise error(msg if msg is not None else 'This pipeline is not and would not be executing. Drop at this point.')


# ------------------ // Logging Section // ------------------
# LOGGING: https://docs.python.org/3/library/logging.html
FORMAT = 'DATETIME: %(asctime)s - TYPE: %(levelno)s >-> MESSAGE: %(message)s'
logging.basicConfig(format=FORMAT)


def _TestStateWithLogging_(condition: bool, base_msg: str, log_function: Callable,
                           msg: Optional[str] = None) -> None:
    if condition:
        return None
    if msg is None:
        msg = base_msg
    return log_function(msg)


def TestStateByWarning(condition: bool, msg: Optional[str] = None) -> None:
    BASE_MSG = 'This pipeline is inaccurate but we continue executing.'
    return _TestStateWithLogging_(condition=condition, base_msg=BASE_MSG,
                                  log_function=logging.warning, msg=msg)


def TestStateByInfo(condition: bool, msg: Optional[str] = None) -> None:
    BASE_MSG = 'Something has been raised to inform the user, but message is NOT specified.'
    return _TestStateWithLogging_(condition=condition, base_msg=BASE_MSG,
                                  log_function=logging.info, msg=msg)


def TestStateByError(condition: bool, msg: Optional[str] = None) -> None:
    BASE_MSG = 'This pipeline is inaccurate but we continue executing.'
    return _TestStateWithLogging_(condition=condition, base_msg=BASE_MSG,
                                  log_function=logging.error, msg=msg)


def TestStateByCritical(condition: bool, msg: Optional[str] = None) -> None:
    BASE_MSG = 'This pipeline is inaccurate but we continue executing.'
    return _TestStateWithLogging_(condition=condition, base_msg=BASE_MSG,
                                  log_function=logging.critical, msg=msg)


def ValidateCondition(condition: bool, errno: bool = True, msg: Optional[str] = None) -> bool:
    if not condition:
        TestState(condition, msg=msg, error=NotImplemented) if errno else \
            TestStateByWarning(condition, msg=f'NotImplemented: {msg}')
    return condition


# -------------------------------------------------------------------------------------------------------------------
# [1]: Checking DataType
__CALLER: str = 'Python built-in'
DATA_TYPE_CACHE_CHECK: Dict[str, List] = \
    {'str': [str, f"{__CALLER} string"], "int": [int, f"{__CALLER} integer string"],
     'bool': [bool, f"{__CALLER} boolean"], "float": [float, f"{__CALLER} float"],
     'List': [List, f"{__CALLER} list"], "Tuple": [Tuple, f"{__CALLER} tuple"],
     'Dict': [Dict, f"{__CALLER} dictionary"], "Set": [Set, f"{__CALLER} set"],
     'Slice': [slice, f"{__CALLER} slice"], 'None': [None, f"{__CALLER} NoneType object"],
     "Callable": [Callable, 'method/pipeline'],
     'DataFrame': [pd.DataFrame, 'Pandas DataFrame'], 'Index': [pd.Index, 'Pandas Index'],
     'coo_matrix': [coo_matrix, 'Scipy coo_matrix'], 'spmatrix': [spmatrix, 'Scipy spmatrix'],
     'csc_matrix': [csc_matrix, 'Scipy csc_matrix'], 'csr_matrix': [csr_matrix, 'Scipy csr_matrix'],
     'ndarray': [ndarray, 'Numpy array']}


def InputFastCheck(value: Any, dtype: Optional[str], delimiter: Optional[str] = None) -> bool:
    """ The base implementation of datatype - argument evaluation on common datatype. """
    if dtype is None or 'None' in dtype:
        if value is None:
            return True

    try:
        target = tuple([DATA_TYPE_CACHE_CHECK[key][0] for key in dtype.split(delimiter)]) \
            if delimiter is not None else DATA_TYPE_CACHE_CHECK[dtype][0]

        if isinstance(target, Tuple):
            if None in target:
                if value is None:
                    return True
                return isinstance(value, tuple([checkDtype for checkDtype in target if checkDtype is not None]))
        return isinstance(value, target)
    except (ValueError, KeyError, IndexError, TypeError):
        warning('Unable to check your value properly as basic datatype is unavailable.')
    return False


def InputFullCheck(value: Any, name: str, dtype: Optional[str], delimiter: Optional[str] = None,
                   warning_only: bool = False, fastCheck: bool = True) -> bool:
    """
    Used to check the input argument parameter in a single shot. Return boolean value
    whether it passed the test if :arg:`warning_only`=True; otherwise, raise TypeError.

    Arguments:
    ---------

    value : Any
        The value of argument needed to be checked.
    
    name : str
        The name of value needed for message display.
    
    dtype : dtype
        The data type needed for evaluation. If multiple data type is set, :arg:`delimiter` 
        must not be None.
    
    delimiter : str, optional
        If set, multiple data types will be checked in one calling by a string separation.
    
    warning_only : bool, optional
        If True, no TypeError made; instead warning called. 
    
    fastCheck : bool, optional
        If True, skip some checking. Only used when you type correct input. Default to False.
    
    Returns:
    -------
    
    A boolean to show whether the value passed the test.

    """
    if not InputFastCheck(fastCheck, dtype='bool'):
        raise TypeError(f"Fast Checking should be {DATA_TYPE_CACHE_CHECK['bool'][1]}.")

    if fastCheck:
        if value is None and dtype is None or dtype.find('None') != -1:
            return True
    else:
        if not InputFastCheck(name, dtype='str'):
            raise TypeError(f"Input Name should be {DATA_TYPE_CACHE_CHECK['str'][1]}.")
        if dtype is not None:
            if not InputFastCheck(dtype, dtype='str'):
                raise TypeError(f"Input Data Type should be {DATA_TYPE_CACHE_CHECK['str'][1]}.")
        elif value is None:  # dtype is None
            return True

        if not InputFastCheck(delimiter, dtype='str') and delimiter is not None:
            raise TypeError(f"Input Delimiter should be {DATA_TYPE_CACHE_CHECK['str'][1]} or NoneType object")
        if not InputFastCheck(warning_only, dtype='bool'):
            raise TypeError(f"warning_only={warning_only} should be {DATA_TYPE_CACHE_CHECK['bool'][1]}")

    outcome: bool = InputFastCheck(value, dtype=dtype, delimiter=delimiter)
    if outcome:
        return outcome
    if dtype.find('-') != -1:
        delimiter = '-'
    target = tuple([DATA_TYPE_CACHE_CHECK[key][0] for key in dtype.split(delimiter)]) \
        if delimiter is not None else DATA_TYPE_CACHE_CHECK[dtype][0]
    msg: str = f" {name} should be {__CALLER} {target} but not type: {type(value)}."

    if warning_only:
        warning(msg)
        return outcome
    raise TypeError(msg)


def _CheckLefty_(value: Union[int, float], minimumValue: Union[int, float], allowBoundary: bool) -> bool:
    return minimumValue <= value if allowBoundary else minimumValue < value


def _CheckRighty_(value: Union[int, float], maximumValue: Union[int, float], allowBoundary: bool) -> bool:
    return maximumValue >= value if allowBoundary else maximumValue > value


def InputCheckRange(value: Union[int, float], name: str, maxValue: Optional[Union[int, float]],
                    minValue: Optional[Union[int, float]] = 0, allowNoneInput: bool = False,
                    allowFloatInput: bool = False, warning_only: bool = False, leftBound: bool = True,
                    rightBound: bool = False) -> bool:
    """
    Used to check the input argument between a range of [minValue, maxValue). Return boolean value
    whether it passed the test if warning_only=True; otherwise, raise TypeError or ValueError.

    Arguments:
    ---------

    value : Any
        The value of argument needed to be checked.
    
    name : str
        The name of value needed for message display.
    
    maxValue : int or float
        The maximum value that the `value` needed to be satisfied. If None, there are 
        no upper-bound.
    
    minValue : int or float, optional
        The minimum value that the `value` needed to be satisfied. If None, there are 
        no lower-bound. Default to zero (:arg:`minValue`=0).
    
    allowNoneInput : bool, optional
        If True, the value can accept to be None. Default to False.
    
    allowFloatInput : bool, optional
        If True, the value can accept to be a float. Otherwise, the value must be a
        python integer. Default to False.
    
    warning_only : bool, optional
        If True, no TypeError or ValueError made; instead warning called. Default to False.
    
    leftBound : bool, optional
        If True, the value can accept to be equal to the minimum value. Default to True.
    
    rightBound : bool, optional
        If True, the value can accept to be equal to the maximum value. Default to False.
    
    Returns:
    -------

    A boolean to show whether the value passed the test.

    """
    if allowNoneInput and value is None:
        return True

    checking_datatype = 'int-float' if allowFloatInput else 'int'
    if minValue is None and maxValue is None:
        warning(f'{name}: Your value is only checked by the datatype. Your input cannot be compared at any metric.')
        return InputFullCheck(value, name=name, dtype=checking_datatype, delimiter='-')

    if maxValue is not None and minValue is not None:
        if minValue > maxValue:
            warning(f'{name}: Input range must be swapped to guarantee consistency.')
            minValue, maxValue = maxValue, minValue

    lBound: str = '(' if minValue is None else ('[' if leftBound else '(')
    rBound: str = ')' if maxValue is None else (']' if rightBound else ')')

    INF: str = 'INFINITE'

    if InputFastCheck(value, dtype=checking_datatype, delimiter='-'):
        msg: str = ''
        if minValue is not None and maxValue is None:
            if not _CheckLefty_(value, minimumValue=minValue, allowBoundary=leftBound):
                msg: str = f' {name}={value} is out-of-range {lBound}{minValue}, {INF}{rBound}'
        elif minValue is None and maxValue is not None:
            if not _CheckRighty_(value, maximumValue=maxValue, allowBoundary=rightBound):
                msg: str = f' {name}={value} is out-of-range {lBound}{INF}, {maxValue}{rBound}'
        elif not (_CheckLefty_(value, minimumValue=minValue, allowBoundary=leftBound) and
                  _CheckRighty_(value, maximumValue=maxValue, allowBoundary=rightBound)):
            msg: str = f' {name}={value} is out-of-range {lBound}{minValue}, {maxValue}{rBound}'

        if msg != '':
            if warning_only:
                warning(msg)
                return False
            raise ValueError(msg)
    else:
        note = 'integer'
        if minValue is not None:
            if minValue >= 0:
                note = 'positive integer'
        elif maxValue is not None:
            if maxValue <= 0:
                note = 'negative integer'

        msg: str = f' {name}={value} must be a {note} {lBound}{minValue}, {maxValue}{rBound}.'
        if allowNoneInput:
            msg = f'{msg} or None'

        if warning_only:
            warning(msg)
            return False
        raise ValueError(msg)
    return True


def InputCheckIterable(value: Union[ndarray, List, Tuple], name: str, maxValue: Optional[Union[int, float]],
                       minValue: Optional[Union[int, float]] = 0, maxInputInside: Optional[int] = 2, **kwargs) -> bool:
    """ Evaluate whether an iterable satisfied to be homogenous within certain range by
        function `InputCheckRange()`. """

    # **kwargs: Argument need for pipeline inputCheckRange
    InputFullCheck(value, name=name, dtype='List-Tuple', delimiter='-')
    InputCheckRange(maxInputInside, name='len(value)', maxValue=None, minValue=0, rightBound=True, allowNoneInput=True)
    if maxInputInside is not None:
        TestState(len(value) <= maxInputInside, f'{name} should have only {maxInputInside} values.')
    return all(InputCheckRange(location, name=f'{name}[{idx}]', maxValue=maxValue, minValue=minValue, **kwargs)
               for idx, location in enumerate(value))


def InputCheckHomogenous(*values, dtype: str, name: str, delimiter: str = None) -> None:
    if delimiter is not None and dtype.find(delimiter) != -1:
        dtypes = dtype.split(delimiter)
    else:
        dtypes = [dtype]
    result = any(all(InputFullCheck(value, name=name, dtype=dtype, delimiter=delimiter,
                                    fastCheck=True) for value in values)
                 for dtype in dtypes)
    if result:
        return
    raise ValueError(f'name={name} contained heterogeneous values, which should be homogenous.')


# -------------------------------------------------------------------------------------------------------------------
# [2]: Decorator and Function used for wrap-up
def MeasureExecutionTime(Function: Callable) -> Callable:
    def compute(*args, **kwargs):
        start: float = perf_counter()
        result = Function(*args, **kwargs)
        print(f'Execution Time ({Function}): {perf_counter() - start:.4f} (s).')
        return result

    return compute


def MemoryProfiler(Object: object, verbose: bool = True, sorting_mode: bool = True,
                   descending: bool = True) -> pd.DataFrame:
    # Hyper-parameter Verification
    InputFastCheck(verbose, dtype='bool')
    InputFastCheck(sorting_mode, dtype='bool')
    InputFastCheck(descending, dtype='bool')

    from sys import getsizeof
    print('=' * 30, MemoryProfiler, '=' * 30)
    total: int = 0
    np_total: int = 0
    arr: List[List[str, str, int]] = []
    for name in Object.__dict__:
        obj = getattr(Object, name)
        size = obj.nbytes if isinstance(obj, ndarray) else getsizeof(obj)
        total += size

        if isinstance(obj, ndarray):
            size = obj.nbytes
            np_total += size
        elif isinstance(obj, (coo_matrix, csc_matrix, csr_matrix, spmatrix)):
            if isinstance(obj, coo_matrix):
                size = obj.data.nbytes + obj.row.nbytes + obj.col.nbytes
            else:
                size = obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes
            np_total += size

        if verbose and not sorting_mode:
            msg = f"{name} ({type(obj)}): \t\t\t\t{size} bytes --> Shape: {obj.shape}" \
                if isinstance(obj, ndarray) else f"{name} ({type(obj)}): \t\t\t\t{size} bytes"
            print(msg)

        arr.append([name, type(obj), size])
    if sorting_mode:
        arr.sort(key=lambda item: int(item[2]), reverse=descending)
    arr: pd.DataFrame = pd.DataFrame(data=arr, index=None, columns=["Name", "Type", "Byte Size"])
    print(arr)

    print("-" * 80)
    percentage: float = np_total / total
    print(f"Attribute Memory: {total} bytes ({round(total / (1024 * 1024), 6)} MB)")
    print(f"Numpy Attribute Memory: {np_total} bytes ({round(np_total / (1024 * 1024), 6)} MB)"
          f" ---> Percentage: {round(100 * percentage, 6)} %")
    print(f"Remaining Memory: {total - np_total} bytes ({round((total - np_total) / (1024 * 1024), 6)} MB) "
          f"---> Percentage: {round(100 * (1 - percentage), 6)} %")
    return arr


def TimingProfiler(Function: Callable):
    def compute(*args, **kwargs):
        from cProfile import Profile
        profiler = Profile()
        profiler.enable()
        result = Function(*args, **kwargs)
        profiler.disable()
        profiler.print_stats(sort=True)
        return result

    return compute
