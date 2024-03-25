# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray

from Bit2Edge.utils.verify import InputFullCheck, TestState, TestStateByWarning
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spmatrix


# [1]: Constant
_LABEL = Optional[Union[ndarray, List[str], pd.Index]]
_FEATURE = Optional[Union[ndarray, csr_matrix, coo_matrix, csc_matrix, spmatrix]]
_INFO = Optional[Union[ndarray, List[str]]]
_TARGET = Optional[Union[ndarray, List[float]]]
_FEATURE_SET: Tuple[str, ...] = ('Train', 'Val', 'Test')

FIXED_INFO_NPDTYPE = np.object_
FIXED_INPUT_NPDTYPE = np.uint8
DEFAULT_OUTPUT_NPDTYPE = np.float32

def GetDtypeOfData(request: str = None):
    result = {'label': _LABEL, 'feature': _FEATURE, 'info': _INFO, 'target': _TARGET, 'feature_set': _FEATURE_SET}
    return result.get(request, None) if request is not None else result


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class DataBlock:
    __slots__ = ('_name', '_cols', '_data')

    def __init__(self, name: str):
        InputFullCheck(name, name='name', dtype='str', delimiter=None)
        self._name: str = name
        self._cols: _LABEL = None
        self._data: Dict[str, Union[_INFO, _TARGET, _FEATURE]] = {key: None for key in _FEATURE_SET}

    def SetColumns(self, cols: _LABEL) -> None:
        self._cols = cols

    def GetColumns(self) -> _LABEL:
        return self._cols

    def SetData(self, data: Union[_INFO, _TARGET, _FEATURE], environment: str = _FEATURE_SET[0]) -> None:
        InputFullCheck(environment, name='environment', dtype='str', delimiter=None)
        accepted_feature = GetDtypeOfData('feature_set')
        TestState(environment in accepted_feature,
                  msg=f'The feature type must be one of these values: {accepted_feature}')
        self._data[environment] = data

    def InitData(self, shape: Tuple[int, ...], dtype, environment: str = _FEATURE_SET[0]) -> None:
        # Reset data, and check the input
        self.SetData(data=None, environment=environment)

        # Initialize data
        data = np.zeros(shape=shape, dtype=dtype)
        self.SetData(data=data, environment=environment)

    def GetData(self, environment: str) -> Union[_INFO, _TARGET, _FEATURE]:
        InputFullCheck(environment, name='environment', dtype='str', delimiter=None)
        accepted_feature = GetDtypeOfData('feature_set')
        TestStateByWarning(environment in accepted_feature,
                           msg=f'The feature type must be one of these values: {accepted_feature}')
        return self._data.get(environment, None)

    def ClearData(self) -> None:
        self._cols = None
        self._data = {key: None for key in _FEATURE_SET}
