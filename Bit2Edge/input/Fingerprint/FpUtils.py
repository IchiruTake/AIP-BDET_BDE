# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module is to serve as the general interface to the RDKit C++ Fingerprint-
# -Generator -> To be the Abstract Class
# --------------------------------------------------------------------------------

from typing import List, Optional, Tuple, Union

from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs import ExplicitBitVect, SparseBitVect

from Bit2Edge.utils.helper import InputCheckRange, InputFullCheck
from Bit2Edge.utils.verify import TestState


def _GetSpecialValue_(value: Union[int, bool, float, str, Tuple], index: int,
                      isRadicalEnv: bool) -> Union[int, bool, float, Tuple]:
    if isinstance(value, (int, float, bool, str)):
        return value

    try:
        return value[index]
    except (IndexError, ValueError) as err:
        if isRadicalEnv and index == len(value) - 1:
            return value[-1]
        raise err


class BVAbstractEngine:
    __slots__ = ('_nBits', '_dist', '_chiral', '_seed', '_cache')

    def __init__(self, nBits: int, dist: int, chiral: bool = False, seed: int = 0, **kwargs):
        self._nBits: int = nBits
        self._dist: int = dist

        self._chiral: bool = chiral
        self._seed: int = seed

        self._cache: dict = kwargs
        self._VerifyConfiguration_()

    def _VerifyConfiguration_(self) -> None:
        InputFullCheck(self._chiral, name='Chirality', dtype='bool')
        InputCheckRange(self._seed, name='Random Seed', maxValue=None, minValue=0)

        InputCheckRange(self._nBits, name='nBits', maxValue=BVAbstractEngine._GetMaxBits(), minValue=0)
        InputCheckRange(self._dist, name='Distance', maxValue=BVAbstractEngine._GetMaxDist(), minValue=0)

    def GetFpBits(self) -> int:
        return self._nBits

    def __len__(self) -> int:
        return self.GetFpBits()

    def GetFpDist(self) -> int:
        return self._dist

    def GetFpStatus(self) -> bool:
        status: bool = self._cache.get('status', None)
        if status is not None:
            InputFullCheck(status, name='status', dtype='bool')
            TestState((not status and self._nBits != 0) or (status and self._nBits == 0),
                      'Conflict state between nBits and status.')
            return status
        return self._nBits != 0

    def GetBaseProperties(self) -> tuple:
        return self._nBits, self._dist, self._chiral, self._seed

    def GetOtherProperties(self) -> dict:
        return self._cache

    def GetFpStringTemplate(self) -> str:
        raise NotImplementedError

    def GetFpLabel(self, start: int = 0) -> List[str]:
        template: str = self.GetFpStringTemplate()
        return [f'{template}{idx}' for idx in range(start, start + self.GetFpBits())]

    def MolToBitVect(self, mol: Mol) -> Optional[Union[ExplicitBitVect, SparseBitVect]]:
        raise NotImplementedError

    @staticmethod
    def _GetMaxDist() -> int:
        return 8

    @staticmethod
    def _GetMaxBits() -> int:
        return 2 ** 16

    def __eq__(self, other):
        if other is self:
            return True
        if not isinstance(other, self.__class__):  # Enforce fixed class only
            raise TypeError(f'Incorrect comparison between {self.__class__} and {type(other)}.')

        if self.GetBaseProperties() != other.GetBaseProperties():
            return False

        return self.GetOtherProperties() == other.GetOtherProperties()
