# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from typing import List, Tuple, Union

from numpy import ndarray

from Bit2Edge.dataObject.InputState import InputState
from Bit2Edge.utils.verify import TestState, InputFullCheck


def _CastLabels_(labels: Union[List[str], ndarray]) -> Union[object, List[str]]:
    InputFullCheck(labels, name='labels', dtype='List-ndarray', delimiter='-')
    if isinstance(labels, ndarray):
        if labels.ndim != 1:
            labels = labels.ravel()
        labels = labels.tolist()
    return labels


def EvalFingerprintLabels(labels: Union[List[str], ndarray]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """ 
    This function is to evaluate all fingerprint positions used in the Bit2Edge project. 
    
    Arguments:
    ---------

    labels : Union[List[str], ndarray]
        The labels of the fingerprint. It can be either a list of string or a numpy array.
    
    Returns:
    -------

    A tuple of two tuples, which contains the start and end index of the fingerprint.

    """
    labels = _CastLabels_(labels)

    # [1]: Initialization
    numsInput, names = InputState.GetNumsInput(), InputState.GetNames()
    ConnectionStorage: List[str] = []
    StartIndex: List[int] = [-1] * numsInput
    EndIndex: List[int] = [0] * numsInput

    # [2]: Finding Fingerprint Location
    BitVectIdentifier: Tuple = (':', 1, 3)
    for index, value in enumerate(labels):
        identifier_position: int = str(value).find(*BitVectIdentifier)
        if identifier_position == -1:  # Break the loop if we cannot find any value (-> Fp should be on the left)
            TestState(index != 0, 'No identifier can be found at the first value. Please check your codeflow.')
            break

        notions: str = str(value)[:identifier_position]
        TestState(notions in names, f'The provided notation (={notions}) is incompatible with {names}.')

        if len(ConnectionStorage) == 0:
            ConnectionStorage.append(notions)
        elif notions != ConnectionStorage[-1]:
            ConnectionStorage.append(notions)

        indexLocation: int = len(ConnectionStorage) - 1
        if StartIndex[indexLocation] == -1:
            StartIndex[indexLocation] = index
            EndIndex[indexLocation] = index
        EndIndex[indexLocation] += 1
    return tuple(StartIndex), tuple(EndIndex)
