# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module attempted to provide a standardized way to control feature input
# into the LBICreator.
# --------------------------------------------------------------------------------

from typing import List, Tuple, Union, Optional

from Bit2Edge.config.userConfig import DATA_FRAMEWORK as dFramework


class LBI_Feat:
    __slots__ = ('_labels', '_error')

    def __init__(self, labels: Union[List[str], Tuple[str, ...]], skip_error: bool = False):
        self._labels: List[str] = list(labels)
        self._error: Tuple = (ValueError, IndexError, KeyError) if not skip_error else None

    def GetLabel(self) -> List[str]:
        return self._labels

    def __len__(self) -> int:
        return len(self.GetLabel())

    def AtomBondToMol(self, NodeOrEdge, stack: List[int], isNeighbor: Optional[bool], index: int,
                      positional_datamap=None, length_datamap=None) -> None:
        raise NotImplementedError('This class should not be instantiated and should be implemented on sub-class.')


class AtomSymbol(LBI_Feat):
    __slots__ = ('_labels', '_error', '_unkAtom')

    def __init__(self, isNeighbor: bool):
        labels = dFramework['Core-Atoms'][0] if not isNeighbor else dFramework['Neighbor-Atoms'][0]
        super(AtomSymbol, self).__init__(labels=labels)
        self._unkAtom: Optional[int] = None
        self._TestSymbol_(isNeighbor)

    def _TestSymbol_(self, isNeighbor: bool):
        key: str = 'Neighbor-Atoms' if isNeighbor else 'Core-Atoms'
        if key in dFramework and dFramework[key][1] is not None:
            self._unkAtom: str = self._labels[(dFramework[key][1])]
        return None

    def AtomBondToMol(self, NodeOrEdge, stack: List[int], isNeighbor: bool, index: int,
                      positional_datamap=None, length_datamap=None):
        try:
            stack[positional_datamap[index][NodeOrEdge.GetSymbol()][isNeighbor]] += 1
        except self._error:
            if self._unkAtom is not None:
                location = positional_datamap[index][self._unkAtom]
                stack[location[0 if len(location) == 1 else isNeighbor]] += 1
        return None
