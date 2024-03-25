# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module is to serve as the supporter interface to resolve the C++ bottleneck
# in the RDKit, especially caching the data that is not naive to the Python.
# --------------------------------------------------------------------------------
#  [WARNING] During the RDKit profiling, it was found that `mol.GetAtoms()` and
#  `mol.GetBonds()` return a `_ROAtomSeq` and a `_ROBondSeq`. Since this is not
#  native by Python, even retrieving variable, the best performance option is to
#  cast the result to `PyObject_Tuple`.
#  However, although the retrieving of `atom.GetBonds()` return a Python tuple
#  which is already have fast looping in python despite having O(N) time complexity
#  instead of O(1) as two functions above.
#  In summary:
#  - `mol.GetAtoms()` and `mol.GetBonds()` have O(1) time in RDKit C++, but looping
#     in Python is extremely costly by for-loop
#  - `atom.GetBonds()` have O(N) time in RDKit C++, but the looping in Python is
#     cheaper instead when calling in Python.
# --------------------------------------------------------------------------------


from typing import Callable

from rdkit.Chem.rdchem import Mol, Atom, Bond, BondType
from Bit2Edge.molUtils.molUtils import PyGetAtoms

def IsNotHydrogen(atom: Atom) -> bool:
    """
    This function is to check if the provided atom is not hydrogen, which is equivalent as `atom.GetAtomicNum() != 1`.

    Arguments:
    ---------

    atom : Atom
        The atom need to be considered.

    Returns:
    -------
    Return True if the molecule is not a hydrogen atom.

    """
    return atom.GetAtomicNum() != 1


# -------------------------------------------------------------------------
# [4]: Molecule Evaluation
# Part 01: Neighbor Looping
def _EvalOxyForNeighborLoop(atom: Atom, bondType: BondType) -> bool:
    charge = atom.GetFormalCharge()
    if charge == 0:  # Only connect with single/double bond
        return bondType != 2  # rdBondType.DOUBLE
    elif charge == -1:
        return True
    return True


def _EvalNitroForNeighborLoop(atom: Atom, bondType: BondType) -> bool:
    charge: int = atom.GetFormalCharge()
    if charge == 0:
        return bondType != 3
    elif charge == -1:
        return bondType != 2
    return True


# 128 elements
AtomicFunction = [True] * 128
AtomicFunction[1] = False
AtomicFunction[7] = _EvalNitroForNeighborLoop
AtomicFunction[8] = _EvalOxyForNeighborLoop
AtomicFunction = tuple(AtomicFunction)


def EvalAtomForNeighborLoop(atom: Atom, bond: Bond, quick: bool = True) -> bool:
    """
    This function is particularly useful as it can be applied in many function that need
    neighbor traversing. Logical operation: NOT condition. Ruled by valence theory
    Example (1): in function _FindEnvWith..._ to prevent looping if condition is not fulfilling.
    If True, the atom may definitely have neighbors to perform looping (Most frequent atom first).
    Link: https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html

    Returns:
    -------
    If True, the atom may have neighbors to perform looping. If False, the atom certainly does not have
    neighbors to perform looping. This condition is based on valence theory but not applicable to every
    possible molecules. For example [H3] (triatomic hydrogen).

    """
    if quick:
        return IsNotHydrogen(atom)

    result = AtomicFunction[atom.GetAtomicNum()]
    if isinstance(result, bool):
        return result
    bType: BondType = bond.GetBondType()
    # Bond-Type (Search Scope): bType > 4 or bType == 0
    return (bType > 4 or bType == 0) or (isinstance(result, Callable) and result(atom, bType))


def IsNotObsoleteAtom(mol: Mol, atom: Atom, bond: Bond, safe: bool = False, quick: bool = True) -> bool:
    """
    This function is to predict that do we need extra traversal. If :arg:`mol` is not cached, we may
    perform traversal (True). If :arg:`mol` is cached and there is more one bond connected, we may
    perform traversal (True). If only one or no bond connected to the provided atom, we skipped the
    checking.

    Arguments:
    ---------

    mol : Mol
        The molecule needed to be considered

    atom : Atom
        The atom needed to be considered

    bond : Bond
        The bond needed to be considered. Only applied if the molecule is not internally cached.

    safe : bool
        If True, the traversal would be done by any means. Default to be False.

    quick : bool
        If True, and the molecule is not internally cached, we predict the state (Apply most for small
        molecules) used . Default to be True.

    Returns:
    -------
    If True, we need to run the traversal. Otherwise, the traversal could be skipped.
    """
    # If we want to do the traversal for universal graph
    if safe:
        return True

    # The connectivity of the molecule is cached, use it to obtain the result
    if hasattr(mol, 'PyEdgeSizeGraph'):
        return mol.PyEdgeSizeGraph[atom.GetIdx()] > 1

    if hasattr(mol, 'PyEdgeGraph'):
        return len(mol.PyEdgeGraph[atom.GetIdx()]) > 1

    # The molecule is not cached, check by the current state to see if there is any possibilities
    # we have more than one bond connected (atom is not the final leaf of the graph).
    # :arg:`quick` should be True.
    #
    return EvalAtomForNeighborLoop(atom, bond, quick)


# -------------------------------------------------------------------------
# Part 02: Stable Looping Looping
OuterElec = (0, 1, 2,
             1, 2, 3, 4, 5, 6, 7, 8,
             1, 2, 3, 4, 5, 6, 7, 8,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8,
             1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)


def GetNOuterElecsAtStable(AtomicNum: int) -> int:
    # Ref: https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/atomic_data.cpp
    return OuterElec[AtomicNum]


def IsTransitionMetal(AtomicNum: int) -> int:
    if AtomicNum <= 20:
        return False
    return 21 <= AtomicNum <= 30 or 39 <= AtomicNum <= 48 or 57 <= AtomicNum <= 80 or 89 <= AtomicNum <= 112


def GetNOuterElectrons(atom: Atom) -> int:
    return GetNOuterElecsAtStable(atom.GetAtomicNum()) + atom.GetExplicitValence() - atom.GetFormalCharge()


def IsOctetValenceStableAtom(atom: Atom) -> bool:
    AtomicNum: int = atom.GetAtomicNum()
    if AtomicNum == 1 or AtomicNum == 2:
        return GetNOuterElectrons(atom) == 2

    return not IsTransitionMetal(AtomicNum) and GetNOuterElectrons(atom) == 8


def IsStableMolecule(mol: Mol, quick: bool = True) -> bool:
    """
    This function is to check whether the molecule is stable or not in the Chemistry field. Return
    True if the input molecule is stable. If :arg:``quick`` is set to be True, the checking was
    only carried out on the hydrogen atom only.

    References:
        1) https://en.wikipedia.org/wiki/Transition_metal
        2) https://en.wikipedia.org/wiki/Octet_rule


    Arguments:
    ---------

    mol : Mol
        The input molecule needs to be considered

    quick : bool
        If True, only hydrogen atoms are tested. Otherwise, all available atoms are in consideration.

    Returns:
    -------
    If True, the input molecule is stable. If False, the input molecule is not stable.

    """
    if quick:
        return all(IsOctetValenceStableAtom(atom) for atom in PyGetAtoms(mol) if not IsNotHydrogen(atom))

    return all(IsOctetValenceStableAtom(atom) for atom in PyGetAtoms(mol))
