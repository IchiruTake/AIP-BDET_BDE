# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module serves as the data getter for the model. The data getter is a function
# which significantly impacts and determines the model's performance and design. 
# Currently, we designed the input of the model as following.
#  - Large Bond Environment (radius=4): Locate at the first index of the input.
#  - Small Bond/Radical Environment(s) (radius=2): ONLY locate at the second index of
#    the input
#  - Localized Bond Information (descriptors): Locate at the third index of the input.
#  - Extremely large bond environment (Optional): Locate at the final index of the
#    input.
# --------------------------------------------------------------------------------


from typing import List, Tuple, Callable, Optional, Any, Dict

from numpy import ndarray
from Bit2Edge.utils.helper import TestState


def _TestState_(state: bool) -> None:
    msg: str = 'This is not the accurate function retriever, or not allowed to be used.'
    return TestState(state, msg=msg, error=ValueError)


def BE_2Input(EnvData: ndarray, LBI_Data: ndarray, Start: Tuple[int, ...], End: Tuple[int, ...],
              *args, **kwargs) -> List[ndarray]:
    """ This is the function retriever used in class B2E_Model. The function is limited
        with only two different radius. """
    _TestState_(len(Start) == 2 and len(End) == 2)
    return [EnvData[:, Start[0]:End[0]], EnvData[:, Start[1]:End[1]], LBI_Data]


def BE_3Input(EnvData: ndarray, LBI_Data: ndarray, Start: Tuple[int, ...], End: Tuple[int, ...],
              *args, **kwargs) -> List[ndarray]:
    """ This is the function retriever used in class B2E_Model. The function is limited
        with only three different radius. """
    _TestState_(len(Start) == 3 and len(End) == 3)
    return [EnvData[:, Start[1]:End[1]], EnvData[:, Start[2]:End[2]], LBI_Data,
            EnvData[:, Start[0]:End[0]], ]


# --------------------------------------------------------------------------------
# shared_mask: The index where reverse masking on array can be applied
__MASK: Dict[Callable, Any] = {
    BE_2Input: {'shared_mask': [0, 1, 2], 'is_all_shared': True, 'differ_func': None, 'lbi': 2},
    BE_3Input: {'shared_mask': [0, 1, 2, 3], 'is_all_shared': True, 'differ_func': None, 'lbi': 2},
}

def _EvalFunc(func: Callable, key: str) -> Optional[Dict]:
    search = __MASK.get(func, None)
    if search is None:
        return None
    return search[key]


def GetSharedMask(func: Callable) -> Optional[List[int]]:
    return _EvalFunc(func, 'shared_mask')


def IsAllShared(func: Callable) -> Optional[bool]:
    return _EvalFunc(func, 'is_all_shared')


def IsDiffFuncAvailable(func: Callable) -> Optional[bool]:
    return _EvalFunc(func, 'differ_func') is not None


def GetDiffFunc(func: Callable) -> Optional[Callable]:
    return _EvalFunc(func, 'differ_func')


def GetLBILocation(func: Callable) -> Optional[int]:
    return _EvalFunc(func, 'lbi')
