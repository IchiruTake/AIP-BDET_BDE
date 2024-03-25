# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# Usage: This module is an optimized example version for function
# `FindAtomEnvironmentOfRadiusN()` of RDKit (version < 2022.03)
# Optimization is done ahead to achieve speed boost
# Note: https://github.com/python/cpython/pull/96350/files
# --------------------------------------------------------------------------------

"""
    Some optimization are done ahead to gain better performance speed-boost in compared to the
    naive implementation which are LastLayerIgnorance, Pre-defined Atomic Valence Traversing,
    and Edge-Graph Caching.
    Given a graph G = {V, E} which have |V| vertices and |E| edges. The number of edges between
    two consecutive layers is |E'| (<< |E|), by the chemical constraint are 6 - 8 edges each.
    On average, assuming half of them are having neighbors to perform extra neighbor searching
    |Vr| and |V~r|. Given the max layer distance is R, in total, the time complexity are:

    ---------------------------------------------------------------
    RDKit (< 2022.03, C++, GetAtomEnvironment): 2(|E| + R * |E'| * |E|)
    RDKit (< 2022.03, C++, GetBondEnvironment, if have): 2|E| + (R + 0.5) * |E'| * |E|
    RDKit (2022.03, C++, GetAtomEnvironment): 2(|E| + (R - 1) * |E'| * |E| + |E'|)
    RDKit (2022.03, C++, GetBondEnvironment, if have): 2|E| + (R - 0.5) * |E'| * |E| + |E'|
    ---------------------------------------------------------------
    Python (Custom Module) - Prefer Molecule Sorting or Arranging:
    Python (GetBondEnvironment, No Caching):
        - Time for one bond: 2|E| + (R - 1)(|E| * |Vr| + |V~r|) + |E'|
        - Time for N bonds: N * (2|E| + (R - 1)(|E| * |Vr| + |V~r|) + |E'|)
        - Memory: O(1)
    Python (GetBondEnvironment, Caching):
        - Time for one bond: |E| + 2|E'| + (R - 1)(|E'| * |Vr| + |V~r|) + |E'|
        - Time for N bonds: |E| + N * (2|E'| + (R - 1)(|E'| * |Vr| + |V~r|) + |E'|)
        - Memory: O(2|E|)
    ---------------------------------------------------------------

    Currently, our RDKit version is 2021.03 so the time different is relatively significant. Assuming
    every constant operation are minimal and costing an equivalent amount of time, we have
    Total (RDKit): 2|E| + 6 * 2 * R * |E| ~= 2|E| + 12 * R * |E|
    Total (Python + Cache): |E| + 2 * 6 + (R - 1)(6 * 3 + 3) + 6 ~= |E| + 12 + 21 * (R - 1) + 6 ~= |E| + 18 + 21 (R - 1)
    Total (Python + No-Cache): 2|E| + (R - 1)(3|E| + 3) + 6 ~= 2|E| + (R - 1)(3|E| + 3) + 6
    For R = 2:
        - RDKit: 2|E| + 24 |E| = 26|E|
        - Python Cache: |E| + 18 + 21 * 1 = |E| + 39
        - Python No-Cache: 2|E| + (3|E| + 3) + 6 = 5|E| + 9
    For R = 4:
        - RDKit: 2|E| + 48 |E| = 50|E|
        - Python Cache: |E| + 18 + 21 * 3 = |E| + 81
        - Python No-Cache: 2|E| + 3(3|E| + 3) + 6 = 11|E| + 15
    For R = 6:
        - RDKit: 2|E| + 72 |E| = 74|E|
        - Python Cache: |E| + 18 + 21 * 5 = |E| + 123
        - Python No-Cache: 2|E| + 5(3|E| + 3) + 6 = 17|E| + 21
"""

from typing import List, Tuple, Dict, Optional, Union

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import PathToSubmol
from Bit2Edge.input.Subgraph import SubgraphBFS, SubgraphFloyd

from Bit2Edge.molUtils.molUtils import PyGetBondWithIdx, Sanitize


def AutoSearchBondEnvs(mol: Mol, radius: Tuple[int, ...], bondIDs: List[int], operation: str = 'BFS',
                       useHs: Union[bool, Tuple[bool, ...]] = True) -> List[List[int]]:
    if operation == 'auto':
        BFSCost = SubgraphBFS.EstimateBFSCost(bondIDs, radius, useHs)['total']
        FloydUseHs = [useHs] * len(radius) if isinstance(useHs, bool) else useHs
        FloydCost = SubgraphFloyd.EstimateFloydCost(mol, bondIDs, FloydUseHs)['total']
        operation = 'BFS' if BFSCost < FloydCost else 'Floyd'
    
    if operation == 'BFS':
        for bondID in bondIDs:
            yield SubgraphBFS.SearchBondEnvs(mol, radius, bondIdx=bondID, useHs=useHs)
    elif operation == 'Floyd':
        SubgraphFloyd.PyFindMinPath(mol)
        if isinstance(useHs, bool):
            for bondID in bondIDs:
                yield [SubgraphFloyd.ExtractFloydResult(mol, bondIdx=bondID, maxValue=r, useHs=useHs) for r in radius]
        else:
            for bondID in bondIDs:
                yield [SubgraphFloyd.ExtractFloydResult(mol, bondIdx=bondID, maxValue=r, useHs=useHs[i])
                       for i, r in enumerate(radius)]
        SubgraphFloyd.CleanupMinPath(mol)
    else:
        raise ValueError('The operation is not supported.')


# ------------------------------------------------------------------------------------
def BondPathToMol(mol: Mol, path: List[int], bondIdx: Optional[int] = None, sanitize: bool = True,
                  atomMap: Dict[int, int] = None) -> Tuple[Mol, Optional[int]]:
    if mol.GetNumConformers() != 0:
        mol.RemoveAllConformers()

    if bondIdx is None:
        smol = PathToSubmol(mol, path, atomMap=atomMap)
        if sanitize:
            Sanitize(smol)
        return smol, None

    if atomMap is None:
        atomMap: Dict[int, int] = {}
    bond = PyGetBondWithIdx(mol, bondIdx)
    smol = PathToSubmol(mol, path, atomMap=atomMap)
    subBondIdx = smol.GetBondBetweenAtoms(atomMap[bond.GetBeginAtomIdx()], atomMap[bond.GetEndAtomIdx()]).GetIdx()
    if sanitize:
        Sanitize(smol)
    return smol, subBondIdx
