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

from Bit2Edge.input.LBI_Feat.LBI_Feat import LBI_Feat


class AtomHybridV1(LBI_Feat):
    def __init__(self):
        labels: List[str] = ['OTHER', 'SP', 'SP2', 'SP3']
        super(AtomHybridV1, self).__init__(labels=labels)

    def AtomBondToMol(self, NodeOrEdge, stack: List[int], isNeighbor: bool, index: int,
                      positional_datamap=None, length_datamap=None):
        try:
            stack[positional_datamap[index][str(NodeOrEdge.GetHybridization())][isNeighbor]] += 1
        except self._error:
            stack[length_datamap[index - 1][isNeighbor]] += 1
        return None


class AtomRingV1(LBI_Feat):
    def __init__(self):
        labels: List[str] = ['Aromatic', 'Ring']
        super(AtomRingV1, self).__init__(labels=labels, skip_error=True)

    def AtomBondToMol(self, NodeOrEdge, stack: List[int], isNeighbor: bool, index: int,
                      positional_datamap=None, length_datamap=None):
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
        elif NodeOrEdge.IsInRing():
            stack[location + 1] += 1
        return None


class BondStateV1(LBI_Feat):
    def __init__(self):
        labels: List[str] = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        super(BondStateV1, self).__init__(labels=labels, skip_error=True)

    def AtomBondToMol(self, NodeOrEdge, stack: List[int], isNeighbor: bool, index: int,
                      positional_datamap=None, length_datamap=None):
        bType = str(NodeOrEdge.GetBondType())
        stack[positional_datamap[index][bType]] += 1
        if NodeOrEdge.GetIsAromatic():
            stack[positional_datamap[index]['SINGLE']] += 1
        return None
