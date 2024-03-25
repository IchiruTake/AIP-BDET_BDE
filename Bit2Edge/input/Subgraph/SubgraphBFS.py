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

from typing import List, Tuple, Dict, Optional, Union

from rdkit.Chem.rdchem import Mol

from Bit2Edge.molUtils.molUtils import PyGetBondNeighbors, IsNotHydrogen, PyGetBondWithIdx, PyGetAtomWithIdx, \
    IsNotObsoleteAtom, PyGetBondNeighborsFilter
from Bit2Edge.utils.verify import TestState

_MAP = Dict[int, int]
_OPMAP = Optional[Dict[int, int]]

# ------------------------------------------------------------------------------------
def _PrepNbrStack(mol: Mol, atomIdx: int, nbrStack: List[Tuple], useHs: bool) -> List[Tuple]:
    """
    This function is to prepare the neighbor stack for the `FindEnvOfRadiusN()` function.

    Arguments:
    ---------

    mol : Mol
        The RDKit molecule

    atomIdx : int
        The atom id for neighbor searching

    nbrStack : List[Tuple]
        The list used to store the result

    useHs : bool
        Whether to consider H-based bond.

    Returns:
    ------=
    The list contained the tuple of neighboring (address of neighborStack).

    """
    atom = PyGetAtomWithIdx(mol, atomIdx)
    for bond in PyGetBondNeighbors(mol, atom):
        if useHs or IsNotHydrogen(bond.GetOtherAtom(atom)):
            nbrStack.append((atomIdx, bond.GetIdx(), bond))
    return nbrStack


def _FindEnvOfRadiusN(mol: Mol, radius, nbrStack: List[Tuple], bondMap: _MAP, useHs: bool,
                      atomMap: _OPMAP = None) -> Tuple[int, List[int], List[int]]:
    """
    The algorithm perform Breadth-First Search on bond.

    Arguments:
    ---------

    mol : Mol
        The RDKit molecule to be considered

    radius : int
        The radius of the sub-graphs to be considered

    nbrStack : List[Tuple]
        The list used to store the result

    bondMap : Dict[int, int]
        The set of bond we used for tracking visited bond path.

    useHs : bool
        Whether to consider H-based bond

    atomMap : Dict[int, int], optional
        Optional, the set of atoms it will measure the minimum distance of the atom
        (atomIdx: minDist).

    safe : bool, optional
        This argument is used to determine whether the traversal should be run or not.
        This argument is used on the `PyIsNextStepTraversalIfNotBranch()` function.

    quick : bool, optional
        This argument is used to determine whether the traversal should be run or not.
        This argument is used on the `PyIsNextStepTraversalIfNotBranch()` function.

    Returns:
    -------

    An integer representing the maximal traversing distance
    The ID list contained the bond path.
    The index list to run the filter based on the distance from the central.
    """
    IsAtomMapNotNone: bool = atomMap is not None
    distances: List[int] = [len(bondMap)]
    for dist in range(1, radius + 1, 1):
        if len(nbrStack) == 0:
            return dist - 1, [*bondMap], distances

        if dist < radius:
            nextLayer: List[Tuple] = []
            for atomIdx, bondIdx, bond in nbrStack:
                if bondIdx not in bondMap:
                    bondMap[bondIdx]: int = dist
                    nextAtomIdx: int = bond.GetOtherAtomIdx(atomIdx)

                    # or dist < atomMap[nextAtomIdx]: Valid condition but never been used as dist is incremental
                    if IsAtomMapNotNone and nextAtomIdx not in atomMap:
                        atomMap[nextAtomIdx] = dist

                    nextAtom = PyGetAtomWithIdx(mol, nextAtomIdx)
                    # A simpler scenario: nextAtom.GetAtomicNum() != 1:
                    if IsNotObsoleteAtom(mol, atom=nextAtom):
                        for nextBond, nextBondIdx in PyGetBondNeighborsFilter(mol, nextAtom, mapping=bondMap):
                            if useHs or IsNotHydrogen(nextBond.GetOtherAtom(nextAtom)):
                                nextLayer.append((nextAtomIdx, nextBondIdx, nextBond))

                        # for nextBond in PyGetBondNeighbors(mol, nextAtom):
                        #     nextBondIdx: int = nextBond.GetIdx()
                        #     if nextBondIdx not in bondMap:
                        #         if useHs or IsNotHydrogen(nextBond.GetOtherAtom(nextAtom)):
                        #             nextLayer.append((nextAtomIdx, nextBondIdx, nextBond))
            nbrStack = nextLayer
        else:
            for atomIdx, bondIdx, bond in nbrStack:
                if bondIdx not in bondMap:
                    bondMap[bondIdx]: int = dist
                    nextAtomIdx: int = bond.GetOtherAtomIdx(atomIdx)
                    if IsAtomMapNotNone and nextAtomIdx not in atomMap:
                        atomMap[nextAtomIdx] = dist

        distances.append(len(bondMap))

    return radius, [*bondMap], distances

def _SetAtomMap(atomMap: _OPMAP, atomIdx1: int, atomIdx2: Optional[int] = None) -> _OPMAP:
    if atomMap is None:
        return None
    if isinstance(atomMap, dict):
        atomMap.clear()
        atomMap[atomIdx1] = 0
        if atomIdx2 is not None:
            atomMap[atomIdx2] = 0
    return atomMap

def _SetBondMap(bondMap: _OPMAP, bondIdx: Optional[int]) -> _MAP:
    if bondMap is None:
        return {bondIdx: 0} if bondIdx is not None else {}
    bondMap.clear()
    if bondIdx is not None:
        bondMap[bondIdx] = 0

    return bondMap

def _LegacyCleanup(atomMap: _OPMAP, bondPath: List[int], bondMap: _MAP) -> None:
    if isinstance(atomMap, dict):
        atomMap.clear()
    bondPath.clear()
    bondMap.clear()

# ------------------------------------------------------------------------------------
def _PySearchAtomEnv(mol: Mol, radius: int, atomIdx: int, useHs: bool = False, enforceSize: bool = True,
                     atomMap: _OPMAP = None, bondMap: _OPMAP = None) -> List[int]:
    # [1]: Initialization
    atomMap = _SetAtomMap(atomMap, atomIdx1=atomIdx)
    bondMap = _SetBondMap(bondMap, None)

    # [2]: Perform Bread-First Search
    neighborStack = _PrepNbrStack(mol, atomIdx, [], useHs)
    maxRadius, bondPath, _ = _FindEnvOfRadiusN(mol, radius, neighborStack, bondMap, useHs, atomMap)

    # [3]: Legacy work -> Can be removed
    if enforceSize and maxRadius != radius:
        _LegacyCleanup(atomMap, bondPath, bondMap)
    return bondPath


def _PySearchBondEnv(mol: Mol, radius: int, bondIdx: int, useHs: bool = False, enforceSize: bool = False,
                     atomMap: _OPMAP = None, bondMap: _OPMAP = None) -> List[int]:
    # [1]: Initialization
    coreBond = PyGetBondWithIdx(mol, bondIdx)
    beginAtomIdx = coreBond.GetBeginAtomIdx()
    endAtomIdx = coreBond.GetEndAtomIdx()

    atomMap = _SetAtomMap(atomMap, atomIdx1=beginAtomIdx, atomIdx2=endAtomIdx)
    bondMap = _SetBondMap(bondMap, bondIdx)

    # [2]: Prepare neighbor stack
    neighborStack = _PrepNbrStack(mol, beginAtomIdx, [], useHs)
    _PrepNbrStack(mol, endAtomIdx, neighborStack, useHs)
    maxRadius, bondPath, _ = _FindEnvOfRadiusN(mol, radius, neighborStack, bondMap, useHs, atomMap)

    # [3]: Legacy work -> Can be removed
    if enforceSize and maxRadius != radius:
        _LegacyCleanup(atomMap, bondPath, bondMap)
    return bondPath


def _PySearchBondEnvs(mol: Mol, radius: Tuple[int, ...], bondIdx: int, useHs: bool = True, enforceSize: bool = False,
                      atomMap: List[_OPMAP] = None, bondMap: List[_OPMAP] = None) -> List[List[int]]:
    """ Find multiple bond paths in the same time, assuming the 'radius' argument is reversely sorted. """
    # [1]: Run algorithm
    # [1.1]: Prepare
    if bondMap is None:
        bondMap: List[_OPMAP] = [None] * len(radius)
    else:
        TestState(len(bondMap) == len(radius), 'Not equivalent bondMap.')
    bondMap[0] = _SetBondMap(bondMap[0], bondIdx)

    coreBond = PyGetBondWithIdx(mol, bondIdx)
    beginAtomIdx = coreBond.GetBeginAtomIdx()
    endAtomIdx = coreBond.GetEndAtomIdx()

    if atomMap is not None:
        TestState(len(atomMap) == len(radius), 'Not equivalent atomMap.')
        for i in range(0, len(radius)):
            atomMap[i] = _SetAtomMap(atomMap[i], atomIdx1=beginAtomIdx, atomIdx2=endAtomIdx)
    else:
        atomMap = [None] * len(radius)

    if radius[0] != max(radius):
        raise ValueError('The radius is not reversely sorted.')

    # [1.2]: Generate feature
    result: List[Optional[List[int]]] = [None] * len(radius)
    stack = _PrepNbrStack(mol, beginAtomIdx, [], useHs)
    _PrepNbrStack(mol, endAtomIdx, stack, useHs)
    dist, result[0], distances = _FindEnvOfRadiusN(mol, radius[0], stack, bondMap[0], useHs, atomMap[0])

    # [1.3]: Legacy work -> Can be removed
    if enforceSize and dist != radius[0]:
        _LegacyCleanup(atomMap[0], result[0], bondMap[0])
        return result

    # [2]: Infer result from the first radius
    num_distances = len(distances)
    for idx, sRadius in enumerate(radius[1:], start=1):
        # Optional
        if isinstance(atomMap[idx], dict):
            aMapDest = atomMap[idx]
            for aIdx, dist in atomMap[0].items():
                if dist > sRadius:
                    break
                aMapDest[aIdx] = dist

        if isinstance(bondMap[idx], dict):
            bMap = bondMap[idx - 1] if isinstance(bondMap[idx - 1], dict) else bondMap[0]
            bMapDest = bondMap[idx]
            for bIdx, dist in bMap.items():
                if dist > sRadius:
                    break
                bMapDest[bIdx] = dist

        if sRadius < num_distances:
            result[idx] = result[0][:distances[sRadius]]
        else:  # distances[sRadius] is not available (molecule is too small)
            result[idx] = result[0]  # result[idx] = result[0].copy()
    return result


def SearchBondEnvs(mol: Mol, radius: Tuple[int, ...], bondIdx: int,
                   useHs: Union[bool, Tuple[bool, ...]] = True) -> List[List[int]]:
    if isinstance(useHs, bool):
        return _PySearchBondEnvs(mol, radius, bondIdx, useHs, atomMap=None, bondMap=None)

    if len(useHs) != len(radius):
        raise ValueError('The length of useHs is not equivalent to the length of radius.')

    temp_paths = _PySearchBondEnvs(mol, radius, bondIdx, useHs=True, atomMap=None, bondMap=None)
    paths = []
    for i, useH in enumerate(useHs):
        p = [bIdx for j, bIdx in enumerate(temp_paths[i]) if not mol.PyHydroBonds[bIdx] and j != 0] \
            if not useH else temp_paths[i]
        paths.append(p)

    return paths

def EstimateBFSCost(bondIDs: List[int], radius: Tuple[int, ...],
                    useHs: Union[bool, Tuple[bool, ...]] = True) -> Dict[str, float]:
    result = {
        'BFS-one': 12.5,
        'BFS-total': 12.5 * max(radius),
        'filter-one-request': 2.75,
    }
    _coef: float = 1.0
    result['BFS-total'] = result['BFS-one'] * max(radius) + result['filter-one-request'] / 2 * (len(radius) - 1)
    result['total'] = result['BFS-total']
    if not isinstance(useHs, bool):
        result['total'] += result['filter-one-request'] * (_coef * useHs.count(False))
    result['total'] = result['total'] * len(bondIDs)
    return result
