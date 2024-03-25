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


from typing import Tuple, Dict, Union, List

from rdkit.Chem.rdchem import RWMol, Mol, Atom, Bond
from rdkit.Chem.rdinchi import InchiToMol
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles, MolFromSmarts
from rdkit.Chem.rdmolops import AddHs, RemoveHs
from rdkit.Chem.rdmolops import Kekulize, SanitizeMol

from Bit2Edge.utils.verify import TestState

_MOL_TYPE = Union[RWMol, Mol]


# --------------------------------------------------------------------------------
# [1]: Molecule-Level
def Sanitize(mol: Mol) -> Mol:
    """ This function is capable of doing in-place operation """
    if mol is not None:
        _ = SanitizeMol(mol, catchErrors=True)  # error
    return mol

def CopyAndKekulize(mol: Mol) -> Mol:
    k_mol: Mol = Mol(mol)
    Kekulize(k_mol, clearAromaticFlags=True)
    return k_mol


def SmilesToSanitizedMol(smiles: str, addHs: bool = True) -> Mol:
    mol = MolFromSmiles(smiles)
    TestState(mol is not None, f'Incorrect SMILES: {smiles}.')
    if addHs:
        mol: Mol = AddHs(mol)
    Sanitize(mol)  # This is to compute the hybridization of hydrogen
    return mol


def SmilesToMol(smiles: str, addHs: bool = True) -> Tuple[Mol, Mol]:
    s_mol = SmilesToSanitizedMol(smiles=smiles, addHs=addHs)
    return s_mol, CopyAndKekulize(s_mol)

def SmilesFromMol(mol: Mol, keepHs: bool = False, maxCanonical: bool = False, **kwargs) -> str:
    result = MolToSmiles(mol if keepHs else RemoveHs(mol), **kwargs)
    if maxCanonical:
        result = MolToSmiles(MolFromSmiles(result))
    return result


def CanonMolString(string: str, mode: str = 'SMILES', useChiral: bool = True) -> str:
    TestState(string is not None and string.count('.') == 0, 'Invalid molecule either None or fragment(s).')
    try:
        if mode in ('SMILES', 'Smiles', 'smiles'):
            return MolToSmiles(MolFromSmiles(string), isomericSmiles=useChiral)
        elif mode in ('InChi', 'inchi', 'Inchi'):
            return MolToSmiles(InchiToMol(string))
        elif mode in ('SMARTS', 'Smarts', 'smarts'):
            return MolToSmiles(MolFromSmarts(string))
    except (RuntimeError, ValueError, TypeError) as e:
        raise e
    raise ValueError('Invalid function for molecule conversion.')


def IsCanonString(string: str, mode: str = 'SMILES', useChiral: bool = True) -> bool:
    try:
        return string == CanonMolString(string, mode=mode, useChiral=useChiral)
    except (RuntimeError, ValueError, TypeError) as e:
        raise e


# --------------------------------------------------------------------------------
# [2]: Bond Type
def ComputeBondType(bond: Bond) -> str:
    StartAtom = bond.GetBeginAtom()
    EndAtom = bond.GetEndAtom()
    return '{}-{}'.format(*sorted([StartAtom.GetSymbol(), EndAtom.GetSymbol()]))


def DetermineBondState(bond) -> Dict:
    state = \
        {
            'aro-ring': False,
            'non-aro-ring': False,
            'aro-ring-att': False,
            'non-aro-ring-att': False,
            'non-ring': False
        }

    if bond.IsInRing():
        aromatic: bool = bond.GetIsAromatic()
        state[('non-' if not aromatic else '') + 'aro-ring'] = True
        return state

    StartAtom = bond.GetBeginAtom()
    EndAtom = bond.GetEndAtom()

    if not StartAtom.IsInRing() and not EndAtom.IsInRing():
        state['non-ring'] = True
        return state

    aromatic: bool = StartAtom.GetIsAromatic() or EndAtom.GetIsAromatic()
    state[('non-' if not aromatic else '') + 'aro-ring-att'] = True
    return state


def DetermineBondStateToString(bond) -> str:
    result = DetermineBondState(bond)
    for k, v in result.items():
        if v:
            return k
    return ''


def DetermineDetailBondType(smiles: str, bondIdx: int, bondType: bool = True) -> str:
    bond_index = int(bondIdx)
    m = SmilesToSanitizedMol(str(smiles), addHs=True)
    BOND = m.GetBondWithIdx(bond_index)

    result: str = f'{ComputeBondType(BOND)}, {DetermineBondStateToString(BOND)}' \
        if bondType else DetermineBondStateToString(BOND)
    if BOND.GetIsConjugated():
        result += ' (c)'
    return result


# --------------------------------------------------------------------------------
# [3]: Atom & Bonds
def PyGetAtoms(mol: _MOL_TYPE, force: bool = False) -> Tuple[Atom, ...]:
    """
    This function is equivalent as `tuple(mol.GetAtoms())` and caching it at the molecule level.
    If the molecule is cached, despite :arg:`zero_cache`=False, the function would reuse that cache.
    If :arg:`force`=True, this function would be recomputed and replaced the cache.

    Arguments:
    ---------

    mol : RWMol or Mol
        The molecule need to be considered.

    force : bool
        If set to True, this function would be recomputed and replaced the cache. Otherwise,
        it would reuse the cache. Default to False.

    Returns:
    -------
    A tuple of RDKit's atoms on that molecule

    """
    # If we cached them already, we exploited the cache
    if not force and hasattr(mol, 'PyAtoms'):
        return mol.PyAtoms

    atoms = tuple(mol.GetAtoms())
    TestState(all(idx == atom.GetIdx() for idx, atom in enumerate(atoms)))
    mol.PyAtoms = atoms
    return atoms


def PyGetAtomWithIdx(mol: _MOL_TYPE, index: int) -> Atom:
    """
    This function is equivalent as `mol.GetAtomWithIdx(index)`. If the molecule is cached,
    the function would reuse that cache.

    Arguments:
    ---------

    mol : RWMol or Mol
        The molecule need to be considered.

    index: int
        The atom's id to consider.

    Returns:
    -------
    The atom with the provided index is returned

    """
    if hasattr(mol, 'PyAtoms'):
        return mol.PyAtoms[index]
    return mol.GetAtomWithIdx(index)


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


def PyGetBonds(mol: Union[Mol, RWMol], force: bool = False) -> Tuple[Bond, ...]:
    """
    This function is equivalent as `tuple(mol.GetBonds())` and caching it at the molecule level.
    If the molecule is cached, despite :arg:`zero_cache`=False, the function would reuse that cache.
    If :arg:`force`=True, this function would be recomputed and replaced the cache.

    Arguments:
    ---------

    mol : RWMol or Mol
        The molecule need to be considered.

    force : bool
        If set to True, this function would be recomputed and replaced the cache. Otherwise,
        it would reuse the cache. Default to False.

    Returns:
    -------
    A tuple of RDKit's bonds on that molecule

    """
    if not force and hasattr(mol, 'PyBonds'):
        return mol.PyBonds
    bonds = tuple(mol.GetBonds())
    TestState(all(idx == bond.GetIdx() for idx, bond in enumerate(bonds)))

    mol.PyBonds = bonds
    mol.PyHydroBonds = [b.GetEndAtom().GetAtomicNum() == 1 or b.GetBeginAtom().GetAtomicNum() == 1 for b in bonds]
    return bonds


def PyGetBondWithIdx(mol: _MOL_TYPE, index: int) -> Bond:
    """
    This function is equivalent as `mol.GetBondWithIdx(index)`. If the molecule is cached,
    the function would reuse that cache.

    Arguments:
    ---------

    mol : RWMol or Mol
        The molecule need to be considered.

    index: int
        The bond's id to consider.

    Returns:
    -------
    The bond with the provided index is returned

    """
    if hasattr(mol, 'PyBonds'):
        return mol.PyBonds[index]
    return mol.GetBondWithIdx(index)


# Caching
def PyCacheEdgeConnectivity(mol: _MOL_TYPE, force: bool = False) -> Tuple[Tuple[Bond, ...], ...]:
    """
    This function attempt to cache all the connectivity of the molecule. If the molecule is cached,
    the function would reuse that cache. If :arg:`force`=True, this function would be recomputed and
    replaced the cache.

    Arguments:
    ---------

    mol : RWMol or Mol
        The molecule need to be considered.

    force : bool
        If set to True, this function would be recomputed and replaced the cache. Otherwise,
        it would reuse the cache. Default to False.

    Returns:
    -------
    A list of tuple/list of RDKit's bonds on that molecule. The tuple/list is the connectivity of the
    atom at index i. The tuple is used if the number of atoms is less than 32, otherwise, the list is used.

    """
    if not force and hasattr(mol, 'PyEdgeGraph'):
        return mol.PyEdgeGraph

    PyAtoms = PyGetAtoms(mol)
    AtomMap = [[] for _ in range(len(PyAtoms))]
    IndexMap = [[] for _ in range(len(PyAtoms))]
    for b_idx, bond in enumerate(PyGetBonds(mol)):
        # Get Index
        b_sidx: int = bond.GetBeginAtomIdx()
        b_eidx: int = bond.GetEndAtomIdx()
        # Fill AtomMap
        AtomMap[b_sidx].append(bond)
        AtomMap[b_eidx].append(bond)
        # Fill IndexMap
        IndexMap[b_sidx].append(b_idx)
        IndexMap[b_eidx].append(b_idx)

    for idx, value in enumerate(AtomMap):
        AtomMap[idx] = tuple(value)
        IndexMap[idx] = tuple(IndexMap[idx])

    mol.PyEdgeGraph = tuple(AtomMap)
    mol.PyEdgeIdxGraph = tuple(IndexMap)
    mol.PyEdgeSizeGraph = tuple([len(imap) for imap in IndexMap])
    return mol.PyEdgeGraph


def PyGetBondNeighbors(mol: _MOL_TYPE, atom: Atom) -> Tuple[Bond, ...]:
    """
    This function attempt to query all neighboring bonds of the atom. If :meth:`PyCacheConnectivity()`
    did not call, it would use the native approach at :arg:`atom`, which is equivalent as `atom.GetBonds()`.

    Arguments:
    ---------

    mol : RWMol or Mol
        The molecule need to be considered.

    atom : Atom
        The atom need to be considered.

    Returns:
    -------

    A tuple/list of RDKit's bonds on that molecule. The tuple/list is the connectivity of the
    atom at index i. The tuple is used if the number of atoms is less than 32, otherwise,
    the list is used.

    """
    if not hasattr(mol, 'PyEdgeGraph'):
        return atom.GetBonds()
    return mol.PyEdgeGraph[atom.GetIdx()]


def PyGetBondNeighborsFilter(mol: _MOL_TYPE, atom: Atom, mapping: Union[dict, list, tuple]) -> Tuple[Bond, int]:
    if not hasattr(mol, 'PyEdgeGraph'):
        PyCacheEdgeConnectivity(mol)

    atIdx: int = atom.GetIdx()
    for bond, bIdx in zip(mol.PyEdgeGraph[atIdx], mol.PyEdgeIdxGraph[atIdx]):
        if bIdx not in mapping:
            yield bond, bIdx

# -------------------------------------------------------------------------
# [4]: Molecule Evaluation
def IsNotObsoleteAtom(mol: Mol, atom: Atom) -> bool:
    """
    This function is to predict that do we need extra traversal. If :arg:`safe` is enabled, we would
    always return True. Otherwise, we would check if the connectivity of the atom is greater than 1.
    Note that this function would force to cache the connectivity of the molecule if it is not cached.

    Arguments:
    ---------

    mol : Mol
        The molecule needed to be considered

    atom : Atom
        The atom needed to be considered

    Returns:
    -------

    - bool:
        Return True if that atom is not obsolete (only one bond connected), otherwise False

    """
    # If the connectivity of the molecule is not cached, we would enable caching for it
    # since PyEdgeGraph is the primary property, other properties did not checked by the function
    if not hasattr(mol, 'PyEdgeGraph'):
        PyCacheEdgeConnectivity(mol) # This would be ensured that the connectivity is cached

    # return len(mol.PyEdgeGraph[atom.GetIdx()]) > 1
    return mol.PyEdgeSizeGraph[atom.GetIdx()] > 1
