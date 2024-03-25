# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module serves as a input-attached class to manage information to feed the
# :cls:`B2E_Model` and its descendent.
# --------------------------------------------------------------------------------


from typing import List, Optional, Tuple, Union

from Bit2Edge.dataObject.DataBlock import FIXED_INPUT_NPDTYPE, DEFAULT_OUTPUT_NPDTYPE


class TrainModelParams:
    __slots__ = ('_InputShape', '_InputDistribution', '_IsInputSparse', '_TargetSize', '_dtype')

    def __init__(self):
        self._InputShape: Optional[List[Tuple[int, int]]] = None
        self._InputDistribution: Optional[Union[List, Tuple]] = None
        self._IsInputSparse: Optional[Union[List[bool], Tuple[bool, ...]]] = None
        self._TargetSize: int = 0
        self._dtype = [FIXED_INPUT_NPDTYPE, DEFAULT_OUTPUT_NPDTYPE]

    def SetInputShape(self, value: Optional[List[Tuple[int, int]]]) -> None:
        self._InputShape = value

    def GetInputShape(self) -> Optional[List[Tuple[int, int]]]:
        return self._InputShape

    def SetInputDistribution(self, value: Optional[Union[List, Tuple]]) -> None:
        self._InputDistribution = value

    def GetInputDistribution(self) -> Optional[Union[List, Tuple]]:
        return self._InputDistribution

    def SetIsInputSparse(self, value: Optional[Union[List, Tuple]]) -> None:
        self._IsInputSparse = value

    def GetIsInputSparse(self) -> Optional[Union[List, Tuple]]:
        return self._IsInputSparse

    def SetTargetSize(self, value: int) -> None:
        self._TargetSize = value

    def GetTargetSize(self) -> int:
        return self._TargetSize

    def SetInputDtype(self, value) -> None:
        self._dtype[0] = value

    def SetOutputDtype(self, value) -> None:
        self._dtype[1] = value

    def GetDtype(self) -> Tuple:
        return tuple(self._dtype)
