# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from typing import Tuple, Optional, List, Union

from Bit2Edge.utils.verify import InputFullCheck, InputCheckRange, TestState, InputCheckIterable


def _IntArgCheck_(x: Optional[int], name: str, useNone: bool = False) -> Optional[int]:
    dtype = 'int' if not useNone else 'int-None'
    InputFullCheck(x, name=name, dtype=dtype, delimiter='-')
    InputCheckRange(x, name=name, maxValue=None, allowNoneInput=useNone, allowFloatInput=False)
    return x


class FileParseParams:
    """
    This object contained pre-defined parameters to read/load/write a pd.DataFrame or np.ndarray
    that is defined by integer-indexed location

    """

    """ This object contained pre-defined parameters to """
    __slots__ = ('_mol', '_r1', '_r2', '_bIdx', '_bType', '_target', '_strict')

    def __init__(self, mol: int = 0, radical: Optional[Tuple[int, int]] = (1, 2), bIdx: Optional[int] = 3,
                 bType: Optional[int] = 4, target: Optional[Union[int, List[int]]] = 5, strict: bool = False,
                 maxSize: Optional[int] = 2 ** 16 - 1):
        # If isStrict = True, all argument (except target) should not be None
        self._mol: int = _IntArgCheck_(mol, name='mol', useNone=not strict)
        self._r1: int = _IntArgCheck_(radical[0], name='r1', useNone=not strict)
        self._r2: int = _IntArgCheck_(radical[1], name='r2', useNone=not strict)
        if (self._r1 is None and self._r2 is not None) or (self._r1 is not None and self._r2 is None):
            raise ValueError('The positions stored as the radical position is not valid.')

        self._bIdx: int = _IntArgCheck_(bIdx, name='bIdx', useNone=not strict)
        self._bType: int = _IntArgCheck_(bType, name='bType', useNone=not strict)

        if target is not None:
            if isinstance(target, int):
                self._target: Optional[List[int]] = [target]
            else:
                self._target: Optional[List[int]] = list(target)
            InputCheckIterable(self._target, name='target', maxValue=2 ** 16 - 1, maxInputInside=None)
        else:
            self._target = None
        self._strict: bool = strict

        self.TestParams(maxSize=maxSize)

    # ----------------------------------------------------------------------------------------------------------------
    def TestParams(self, maxSize: Optional[int] = None) -> None:
        stackv1 = self.DistributeData(useTarget=True, optAll=False)
        TestState(len(set(stackv1)) == len(stackv1), 'Your input cannot contain duplicate.')

        if maxSize is not None:
            InputCheckRange(maxSize, name='maxSize', maxValue=None, minValue=0, allowNoneInput=True)
            TestState(not any(st < 0 or st >= maxSize for st in stackv1),
                      'Invalid argument is found (either negative or exceeded arg::maxsize).')

        if self._strict:
            data = self.DistributeData(useTarget=True, optAll=True)
            TestState(data.count(None) <= 1, 'The parameters contained more than two None values.')
        return None

    def IsEnoughComponent(self, target: bool = False, error: bool = False) -> bool:
        arguments = self.Properties(groupMode=False)
        valid: bool = all(arg is not None for arg in arguments)
        if target:
            valid: bool = valid and self._target is not None
        if not valid and error:
            TestState(False, 'The parameters are not enough to generate a valid file for molecule parsing.')
        return valid

    def DistributeData(self, useTarget: bool, optAll: bool = False) -> List[Optional[int]]:
        stack: List[int] = []
        if optAll or self.Mol() is not None:
            stack.append(self.Mol())
        if optAll or all(radical is not None for radical in self.Radical()):
            stack.extend(self.Radical())
        if optAll or self.BondIndex() is not None:
            stack.append(self.BondIndex())
        if optAll or self.BondType() is not None:
            stack.append(self.BondType())
        if useTarget and (optAll or self.Target() is not None):
            stack.extend(self.Target())
        return stack

    # ----------------------------------------------------------------------------------------------------------------
    def Mol(self) -> int:
        return self._mol

    def Radical(self) -> Tuple[int, int]:
        return self._r1, self._r2

    def BondIndex(self) -> int:
        return self._bIdx

    def BondType(self) -> int:
        return self._bType

    def Target(self) -> List[int]:
        return self._target

    def Properties(self, groupMode: bool = True) -> Tuple:
        if groupMode:
            return self.Mol(), self.Radical(), self.BondIndex(), self.BondType()
        return self.Mol(), self._r1, self._r2, self.BondIndex(), self.BondType()
