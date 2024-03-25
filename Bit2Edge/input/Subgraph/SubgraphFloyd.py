# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
""" This is build under the assumption that the molecule is sanitized and cached. """

import itertools
import math
from typing import List, Dict
from rdkit.Chem.rdchem import Mol

import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from itertools import combinations, accumulate
from scipy.sparse import csr_matrix


def _Filter(mat: np.ndarray, maxValue: int = 6) -> np.ndarray:
    return np.where(mat <= maxValue)[0]


def _ConstructGraph(graph: List[List[int]], n: int):
    """
    This function construct a graph from the connectivity matrix.
    The graph would have the first index as the redundant array to support the itertools.accomulate function.

    Arguments:
    ---------

    graph: List[List[int]]
        The connectivity matrix, extracted from the cached molecule, which is equivalent of (mol.PyEdgeIdxGraph).

    n: int
        The number of bonds in the molecule.

    """
    array = [[] for _ in range(n + 1)]
    for _, connectivity in enumerate(graph):
        for bIdx_i, bIdx_j in combinations(connectivity, 2):
            array[bIdx_i + 1].append(bIdx_j)
            array[bIdx_j + 1].append(bIdx_i)
    len_arr = list(accumulate([len(x) for x in array]))
    flatten_array = list(itertools.chain.from_iterable(array))
    return array, len_arr, flatten_array


def PyFindMinPath(mol: Mol) -> np.ndarray:
    """
    Find distance between two arbitrary bonds in the molecule using Floyd-Warshall algorithm.

    Arguments:
    ---------

    mol: Mol
        The molecule to be processed.

    force: bool
        If True, the function will recalculate the connectivity matrix.

    Returns:
    -------

    mtx: np.ndarray
        The distance matrix.

    """
    n: int = len(mol.PyBonds)
    array, len_arr, flatten_array = _ConstructGraph(graph=mol.PyEdgeIdxGraph, n=n)
    mol.PyConnectMatrix = array

    matrix = csr_matrix((np.array([1] * len(flatten_array), dtype=np.uint8), flatten_array, len_arr), shape=(n, n))
    # The matrix is already unweighted, so we don't need to do anything.
    mtx = floyd_warshall(matrix, directed=True).astype(np.uint32)
    mol.PyMinPath = mtx
    return mtx


def ExtractFloydResult(mol: Mol, bondIdx: int, maxValue: int, useHs: bool = True):
    # Get all the bond index from PyMinPath that is smaller than maxValue.
    result = _Filter(mol.PyMinPath[bondIdx], maxValue).tolist()
    if useHs:
        return result
    # We omit the hydrogen in the calculation, but we have to ensure the index must be in the result, regardless
    # of whether the bond is a hydrogen bond or not. As we used logical_not, and the mol.PyHydroBonds return a
    # list of boolean (True means it is hydrogen bond), we must set the index as False to ensure it is not omitted.
    return [bondpath for bondpath in result if not mol.PyHydroBonds[bondpath] or bondpath == bondIdx]


def CleanupMinPath(mol: Mol) -> None:
    # This function is used to clean up the memory.
    del mol.PyMinPath
    del mol.PyConnectMatrix

def EstimateFloydCost(mol: Mol, bondIDs: List[int], useHs: List[bool]) -> Dict[str, float]:
    n: int = len(mol.PyBonds)
    result = {
        'graph': 0.682 + 0.7275 * n,
        'csr': 41.45 + 0.7755 * n,
        'floyd': 9.7789 + 0.01996 * (n ** 2) + 0.0006 * (n ** 3),
        'filter-one-request': 2.75,
    }
    _coef: float = 1.0
    result['filter-full-request'] = result['filter-one-request'] * (len(useHs) + _coef * useHs.count(False))
    result['filter-total'] = result['filter-full-request'] * len(bondIDs)
    result['total'] = result['graph'] + result['csr'] + result['floyd'] + result['filter-total']
    return result