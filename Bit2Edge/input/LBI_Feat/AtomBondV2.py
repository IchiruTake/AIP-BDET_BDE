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

from typing import List

from rdkit.Chem.rdchem import BondType as rdBondType
from rdkit.Chem.rdchem import HybridizationType as rdHybrid

from Bit2Edge.input.LBI_Feat.LBI_Feat import LBI_Feat


class AtomHybridV2(LBI_Feat):
    def __init__(self):
        labels = [rdHybrid.S, rdHybrid.SP, rdHybrid.SP2, rdHybrid.SP3, (rdHybrid.SP3D, rdHybrid.SP3D2)]
        super(AtomHybridV2, self).__init__(labels=labels)

    def AtomBondToMol(self, NodeOrEdge, stack: List[int], isNeighbor: bool, index: int,
                      positional_datamap=None, length_datamap=None) -> None:
        try:
            stack[positional_datamap[index][NodeOrEdge.GetHybridization()][isNeighbor]] += 1
        except self._error:
            pass
        return None


class AtomRingV2(LBI_Feat):
    def __init__(self):
        labels = ['Aromatic', 'Ring', 'Stable-Ring']
        super(AtomRingV2, self).__init__(labels=labels, skip_error=True)

    def AtomBondToMol(self, NodeOrEdge, stack: List[int], isNeighbor: bool, index: int,
                      positional_datamap=None, length_datamap=None) -> None:
        """
            This function is used to define the ring-state of atom in the fastest mode.
            Since the 'aromatic' property has already been defined, we don't look it back: O(1)
            The aromatic state is based on the four rules of aromaticity.

            Arguments:
            -----------
            - atom: The rdkit atom
            - stack (list): A list of integer where we store value
            - isNeighbor (bool/int): Is this atom the core atom
            - indexInTokenDict (int): The order defined in cls.__init__ of AtomFeatures

            Returns:
            -----------
            - A list of integer. The first argument defined the aromatic state,
              the second argument defined whether atom is in a ring,
              the third argument (optional) whether atom is in a 6-membered ring,

            Note that based on RDKit source code, ring searching will have O(k) time complexity
            where k is the minimum expansion to reach at least the cycle or full atom,
            so we try to prevent these case.
        """
        location: int = length_datamap[index - 1][isNeighbor]
        if NodeOrEdge.GetIsAromatic():
            stack[location] += 1
            stack[location + 1] += 1
            if NodeOrEdge.IsInRingSize(6):
                stack[location + 2] += 1
        elif NodeOrEdge.IsInRingSize(6):
            stack[location + 1] += 1
            stack[location + 2] += 1
        elif NodeOrEdge.IsInRing():
            stack[location + 1] += 1
        return None


class AtomChargeV2(LBI_Feat):
    def __init__(self):
        labels = ['PCharge', 'NCharge', 'NumPCharge', 'NumNCharge']
        super(AtomChargeV2, self).__init__(labels=labels, skip_error=True)

    def AtomBondToMol(self, NodeOrEdge, stack: List[int], isNeighbor: bool, index: int,
                      positional_datamap=None, length_datamap=None) -> None:
        charge: int = NodeOrEdge.GetFormalCharge()
        if charge == 0:
            return None

        BaseLocation = length_datamap[index - 1][isNeighbor]
        if charge < 0:  # Positive charge first, negative charge later
            BaseLocation += 1
            charge = -charge
        stack[BaseLocation] += charge  # Total charge
        stack[BaseLocation + 2] += 1  # Number of atom in charge
        # The maximum charge or minimum charge (not recommended) may be available if data contained
        # more charged bond/molecule/environment
        return None


class _BondTypeV2(LBI_Feat):

    def __init__(self, labels: List):
        # See here: https://www.quora.com/Why-dative-bond-is-shown-by-double-bond
        super(_BondTypeV2, self).__init__(labels=labels)

    def AtomBondToMol(self, NodeOrEdge, stack: List[int], isNeighbor: bool, index: int,
                      positional_datamap=None, length_datamap=None) -> None:
        try:
            stack[positional_datamap[index][NodeOrEdge.GetBondType()][isNeighbor]] += 1
        except self._error:
            pass
        return None


class BondTypeV2_01(_BondTypeV2):
    def __init__(self):
        labels = [rdBondType.SINGLE, (rdBondType.DOUBLE, rdBondType.DATIVE), rdBondType.TRIPLE, rdBondType.AROMATIC]
        super(BondTypeV2_01, self).__init__(labels=labels)


class BondTypeV2_02(_BondTypeV2):
    def __init__(self):
        labels = [rdBondType.SINGLE, rdBondType.AROMATIC, (rdBondType.DOUBLE, rdBondType.DATIVE, rdBondType.TRIPLE)]
        super(BondTypeV2_02, self).__init__(labels=labels)


class BondRingV2(LBI_Feat):
    def __init__(self):
        labels = ['RING', 'STABLE-RING', 'Conjugated']
        super(BondRingV2, self).__init__(labels=labels, skip_error=True)

    def AtomBondToMol(self, NodeOrEdge, stack: List[int], isNeighbor: bool, index: int,
                      positional_datamap=None, length_datamap=None) -> None:
        bType = NodeOrEdge.GetBondType()
        BOND_STATE = positional_datamap[index]
        if bType == 12:  # Code of rdBondType.AROMATIC == 12
            stack[BOND_STATE['RING'][isNeighbor]] += 1
            if NodeOrEdge.IsInRingSize(6):
                stack[BOND_STATE['STABLE-RING'][isNeighbor]] += 1

        elif NodeOrEdge.IsInRingSize(6):
            stack[BOND_STATE['RING'][isNeighbor]] += 1
            stack[BOND_STATE['STABLE-RING'][isNeighbor]] += 1

        elif NodeOrEdge.IsInRing():
            stack[BOND_STATE['RING'][isNeighbor]] += 1

        if NodeOrEdge.GetIsConjugated():
            stack[BOND_STATE['Conjugated'][isNeighbor]] += 1
        return
