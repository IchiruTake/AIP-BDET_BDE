# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from typing import Tuple

from Bit2Edge.config.userConfig import DATA_FRAMEWORK as dFramework
from Bit2Edge.config.userConfig import UpdateDataConfig as UC_UpdateDataConfig
from Bit2Edge.utils.verify import (InputCheckIterable, TestState, TestStateByWarning)


def _RetrieveRadius_() -> Tuple[Tuple[int, ...], bool]:
    """
        This function determining the radius for environment benching.
        Returns:
        -------
            - A tuple or radius reversely sorted
            - A boolean determining whether radical environment is applied.
    """
    if isinstance(dFramework['Radius'], int):
        radius = [dFramework['Radius']]
    else:
        radius = list(dFramework['Radius'])
    InputCheckIterable(radius, name='radius', maxValue=None, minValue=1, maxInputInside=3)
    TestState(len(set(radius)) == len(radius), f'There are a duplicated radius value found: {radius} >> Removed.')
    TestStateByWarning(len(radius) <= 4, 'You attempt to use a too many radius-layer for this problem.')

    radius: list = list(set(radius))
    radius.sort(reverse=True)
    TestStateByWarning(radius[-1] <= 2, 'You attempt to use a too large radius for too small environment.')
    TestStateByWarning(radius[0] <= 6, 'You attempt to use a too large radius for the solving problem.')

    return tuple(radius), False


class InputState:
    # The radius is sorted reversely, and it could be different from the stored configuration
    radius, UseRadicalEnv = _RetrieveRadius_()
    IndexForBondEnv: Tuple[int, ...] = None
    IndexForRadicalEnv: Tuple[int, ...] = None

    @staticmethod
    def ResetInputState() -> None:
        InputState.radius, InputState.UseRadicalEnv = _RetrieveRadius_()
        InputState.IndexForBondEnv = None
        InputState.IndexForRadicalEnv = None

    @staticmethod
    def UpdateState(FilePath: str) -> None:
        UC_UpdateDataConfig(FilePath=FilePath)
        InputState.ResetInputState()

    @staticmethod
    def GetRadius() -> Tuple[int, ...]:
        return InputState.radius

    @staticmethod
    def GetNumsInputByRadiusLayer() -> int:
        result: int = InputState.GetNumsInput()
        # This is checked above, but addition test is recommended
        TestState(result == len(set(InputState.GetRadius())), 'There are more than one radius duplicated.')
        return result

    @staticmethod
    def GetUniqueRadius() -> Tuple[int, ...]:
        return InputState.GetRadius()[0:InputState.GetNumsInputByRadiusLayer()]

    @staticmethod
    def GetIsUseRadicalEnv() -> bool:
        return InputState.UseRadicalEnv

    @staticmethod
    def GetNumsInput() -> int:
        return len(InputState.GetRadius())

    @staticmethod
    def GetStartLocationIfPossibleRadicalEnv() -> int:
        return -1 + InputState.GetNumsInput()

    @staticmethod
    def GetFullNames() -> Tuple[str, ...]:
        # We only needed 3-4 values at most
        return 'X', 'Y', 'Z', 'T', 'A', 'B'

    @staticmethod
    def GetNames() -> Tuple[str, ...]:
        # We only needed 3-4 values at most
        return InputState.GetFullNames()[0:InputState.GetNumsInput()]

    @staticmethod
    def GetIndexForBondEnvironment() -> Tuple[int, ...]:
        if InputState.IndexForBondEnv is None:
            res = list(range(0, InputState.GetNumsInputByRadiusLayer()))
            InputState.IndexForBondEnv = tuple(res)
        return InputState.IndexForBondEnv

    @staticmethod
    def GetIndexForRadicalEnvironment() -> Tuple[int, ...]:
        if InputState.IndexForRadicalEnv is None:
            res = tuple()
            InputState.IndexForRadicalEnv = res
        return InputState.IndexForRadicalEnv
