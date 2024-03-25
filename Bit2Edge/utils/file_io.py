# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from logging import info
from os.path import isfile
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray

from Bit2Edge.config.startupConfig import EXTRA_LIBRARY
from Bit2Edge.utils.verify import InputFastCheck, InputFullCheck, ValidateCondition


def _StringValidation_(FileName: str, extension: str) -> None:
    """ This function is used to check the datatype of a given string """
    InputFullCheck(FileName, name='FileName', dtype='str')
    InputFullCheck(extension, name='extension', dtype='str')


def FixPath(FileName: str, extension: str) -> str:
    """ This function is used to ensure your filename will have the required extension without duplication. """
    _StringValidation_(FileName=FileName, extension=extension)
    return f'{FileName}{extension}' if FileName.rfind(extension) != len(FileName) - len(extension) else FileName


def RemoveExtension(FileName: str, extension: str) -> str:
    """ This function is used to remove the extension out of your filename. """
    _StringValidation_(FileName=FileName, extension=extension)
    return FileName if FileName.rfind(extension) != len(FileName) - len(extension) \
        else FileName[:len(FileName) - len(extension)]


def TestIsValidFilePath(FilePath: str, errno: bool = True) -> bool:
    return ValidateCondition(isfile(FilePath), errno=errno,
                             msg=f'The :arg:`FilePath`={FilePath} is not available.')


def ReadFile(FilePath: Optional[str], header: Optional[int] = 0, dtype=None, get_values: bool = False,
             get_columns: bool = False, use_pandas: bool = False, nrows: Optional[int] = None,
             blocksize: Union[float, int] = 64e6, dtypes_memory_identifier: Union[float, int] = 1,
             usecols: Union[List[int], List[str]] = None, skiprows: Optional[Union[List, int]] = None) \
        -> Optional[Union[pd.DataFrame, List[str], ndarray, Tuple[ndarray, List[str]]]]:
    """
    Default implementation used to call a .csv documentation. Similar when call
    `pd.read_csv` or `dask.read_csv` with higher memory consideration.

    1 MiB = 2^10 KiB = 2^20 bytes = 1048576 bytes
    1 MB = 10^3 KB = 10^6 bytes = 1000000 bytes

    Arguments:
    ---------

    FilePath : str:
        The path contained the .csv file. This argument does not need extension name as it
        must be validated before reading the file.

    header : int
        The row of the file used as the column. Default to 0.

    dtype : dtype
        pandas dtype or numpy.dtype

    get_values : bool
        Whether to get values only

    get_columns : bool
        Whether to get columns only

    use_pandas : bool
        If set to True, we force to use the pandas' library.

    nrows : int
        The number of rows for consideration. Default to None (read all rows)

    skiprows : List or int
        The number of rows we skipped.

    usecols : List or int
        The columns we want to extract. Default to None (read all columns)

    blocksize : int
        The chunking memory for paralleling in Dask Library. Default to be 64 MB (64e6)

    dtypes_memory_identifier : int
        The coefficient memory adding when reading csv by Dask Library. Default to be 1 MiB (1).

    Returns:
    -------

    If :arg:`get_values`=False & :arg:`get_columns`1`=False, an object of pd.DataFrame.
    If :arg:`get_values`=False & :arg:`get_columns`=True, a list of string.
    If :arg:`get_values`=True & :arg:`get_columns`=False, a numpy array.
    If :arg:`get_values`=False & :arg:`get_columns`=False, a tuple of a numpy array and a list of string
    """
    if True:
        if FilePath is None or FilePath == '':
            return None

        InputFullCheck(FilePath, name='FilePath', dtype='str')
        InputFullCheck(get_values, name='get_values', dtype='bool')
        InputFullCheck(get_columns, name='get_columns', dtype='bool')

        if not get_values and get_columns:
            nrows: int = 0

        if not EXTRA_LIBRARY['Dask'] and not EXTRA_LIBRARY['Dask_activated']:
            EXTRA_LIBRARY['Dask_activated'] = True
            try:
                import dask.dataframe as dd
                EXTRA_LIBRARY['Dask'] = True
                info('Dask is a great tool to replicate pandas.DataFrame with read_csv.')
            except (ImportError, ImportWarning):
                info('Dask is not in your environment >> Switch to pandas.')
        pass

    FilePath: str = FixPath(FileName=FilePath, extension=".csv")
    File: Optional[pd.DataFrame] = None
    TestIsValidFilePath(FilePath)
    if not use_pandas and EXTRA_LIBRARY['Dask'] and nrows != 1:
        try:
            import dask.dataframe as dd
            MiB: int = 1048576
            File: pd.DataFrame = \
                dd.read_csv(FilePath, dtype=dtype, header=header, low_memory=True, usecols=usecols, blocksize=blocksize,
                            sample=int(MiB * dtypes_memory_identifier), cache_dates=False).compute()
        except (ValueError, MemoryError, ModuleNotFoundError):
            pass

    if File is None:
        File: pd.DataFrame = pd.read_csv(FilePath, dtype=dtype, nrows=nrows, skiprows=skiprows, usecols=usecols,
                                         header=header, low_memory=True, cache_dates=False)

    if not get_values and not get_columns:
        return File
    if not get_values and get_columns:
        return File.columns.tolist()
    if get_values and not get_columns:
        return File.values if InputFastCheck(File.values, 'ndarray') else np.array(File.values, dtype=dtype)

    return File.values if InputFastCheck(File.values, 'ndarray') else np.array(File.values, dtype=dtype), \
        File.columns.tolist()


def ReadLabelFile(FilePath: Optional[str], header: Optional[int] = 0, dtype=None, nrows: Optional[int] = None,
                  blocksize: Union[float, int] = 64e6, dtypes_memory_identifier: Union[float, int] = 1,
                  usecols: Union[List[int], List[str]] = None,
                  skiprows: Optional[Union[List, int]] = None) -> Union[object, List[str]]:
    return ReadFile(FilePath=FilePath, header=None, get_columns=False, get_values=True, dtype=dtype, use_pandas=True,
                    nrows=nrows, blocksize=blocksize, dtypes_memory_identifier=dtypes_memory_identifier,
                    usecols=usecols, skiprows=skiprows)[header].tolist()


def ExportFile(DataFrame: pd.DataFrame, FilePath: str, index: bool = False,
               index_label: Optional[str] = None) -> None:
    """
    Default implementation used to return the .csv documentation from DataFrame.
    This is similar as `DataFrame.to_csv()`

    Arguments:
    ---------

    FilePath : str
        The path contained the .csv file. This argument does not need extension name as it must be
        validated before reading the file.
    index : bool, List or Tuple
        The implicit array-like used for row indexing (Array-like). Default to False

    index_label : str
        The name of index column. Default to be None.
    """
    if FilePath is None:
        return None
    InputFullCheck(DataFrame, name='DataFrame', dtype='DataFrame')
    DataFrame.to_csv(FixPath(FileName=FilePath, extension='.csv'), index=index, index_label=index_label)


def GetDfValueByInt(DataFrame: pd.DataFrame, column: int) -> ndarray:
    InputFullCheck(DataFrame, name='DataFrame', dtype='DataFrame')
    return DataFrame[DataFrame.columns[column]].values
