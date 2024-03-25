# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module served as an input-controlled parameters for the :meth:`GetBond()` in
# the class :class:`Bit2Edge.dataObject.MolEngine.MolEngine`.
# --------------------------------------------------------------------------------

from typing import List, Tuple, Optional, Union, Any, Dict


class BondParams:
    __slots__ = ('ZeroDuplicate', 'AddReverseReaction',
                 'CoreAtomBondType', 'NeighborAtoms', 'StrictNeighborAtomRule',
                 'IgnoreNonSingle', 'AromaticRing', 'NonAromaticRing', 'AromaticRingAttached',
                 'NonAromaticRingAttached', 'NonRing', 'UseIsotope', 'UseCanonicalOutput')

    def __init__(self):
        # [1]: Unconditional Filter
        self.ZeroDuplicate: bool = True
        self.AddReverseReaction: bool = False

        # [2]: Filter on center-neighbors
        # [2.1]: Filter on atom-level
        self.CoreAtomBondType: Optional[Union[str, Tuple[str, str]]] = None
        self.NeighborAtoms: Optional[List[str]] = None
        self.StrictNeighborAtomRule: bool = False

        # [2.2] Filter on bond-level
        self.IgnoreNonSingle: bool = False
        self.AromaticRing: bool = False
        self.NonAromaticRing: bool = False
        self.AromaticRingAttached: bool = True
        self.NonAromaticRingAttached: bool = True
        self.NonRing: bool = True

        # [3]: Filter on molecule level
        self.UseIsotope: bool = False
        self.UseCanonicalOutput: bool = True

    def GetUnconditionalFiltering(self) -> Dict[str, Any]:
        return {
            'ZeroDuplicate': self.ZeroDuplicate,
            'AddReverseReaction': self.AddReverseReaction,
        }

    def GetAtomFilter(self) -> Dict[str, Any]:
        return {
            'CoreAtomBondType': self.CoreAtomBondType,
            'NeighborAtoms': self.NeighborAtoms,
            'StrictNeighborAtomRule': self.StrictNeighborAtomRule,
        }

    def GetBondFilter(self) -> Dict[str, Any]:
        return {
            'IgnoreNonSingle': self.IgnoreNonSingle,
            'AromaticRing': self.AromaticRing,
            'NonAromaticRing': self.NonAromaticRing,
            'AromaticRingAttached': self.AromaticRingAttached,
            'NonAromaticRingAttached': self.NonAromaticRingAttached,
            'NonRing': self.NonRing,
        }

    def GetMolFilter(self) -> Dict[str, Any]:
        return {
            'UseIsotope': self.UseIsotope,
            'UseCanonicalOutput': self.UseCanonicalOutput
        }

    def GetAllFields(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for func in (self.GetUnconditionalFiltering, self.GetAtomFilter, self.GetBondFilter, self.GetMolFilter):
            result.update(func())
        return result

    def __str__(self):
        return self.GetAllFields()
