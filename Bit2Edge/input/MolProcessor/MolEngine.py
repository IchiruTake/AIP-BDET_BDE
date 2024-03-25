# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module stored some utility functions for bond getter
# --------------------------------------------------------------------------------

import re
from logging import warning
from typing import List, Tuple, Optional, Dict, Union

import pandas as pd
from rdkit.Chem import GetPeriodicTable
from rdkit.Chem.rdchem import (Mol, RWMol)
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import Kekulize, RemoveHs

from Bit2Edge.input.BondParams import BondParams
from Bit2Edge.input.MolProcessor.utils import _LABEL
from Bit2Edge.molUtils.molUtils import IsNotHydrogen, PyGetBonds, Sanitize, IsCanonString, SmilesFromMol, \
    SmilesToSanitizedMol, DetermineBondState, ComputeBondType
from Bit2Edge.utils.helper import ExportFile
from Bit2Edge.utils.verify import InputFullCheck, TestState, TestStateByWarning

# --------------------------------------------------------------------------------
_RE_Compiler: re.Pattern = re.compile(r'\{\d*\}*')
_PERIODIC_TABLE = GetPeriodicTable()


class MolEngine:
    __slots__ = ('_cache',)

    def __init__(self, ):
        self._cache: Dict = {}

    def CleanupCache(self):
        self._cache.clear()

    def GetBond(self, smiles: str, params: Optional[BondParams] = None) -> Tuple[Mol, List[List]]:
        """
        This function is to get all bonds inside molecule (in SMILES form) into a series of bonds.
        The result is a list of reactions, so that each reaction is a list.

        Arguments:
        ---------

        smiles : string
            A molecule in SMILES form
        
        params : BondParams
            The parameters for getting bonds

        Returns:
        -------
        A sanitized molecule and a list of reactions, whose each reaction is a list.

        """
        # [1]: Preparation
        self.CleanupCache()
        if params is None:
            params = BondParams()
        self._BuildAtomBondType_(params.CoreAtomBondType)
        self._BuildNeighborAtomMap_(params.NeighborAtoms)
        mol, (CanonSmiles, IsoSmiles), isIsomeric = self._OptMol_(smiles=smiles, params=params)
        reactions = []
        for bond in PyGetBonds(mol):
            StartAtom = bond.GetBeginAtom()
            EndAtom = bond.GetEndAtom()
            # [1]: Check structure (atom-neighbor-bond type)
            if not self._TestAtomBondType_(StartAtom, EndAtom):
                continue

            if params.IgnoreNonSingle and bond.GetBondType() not in (1, 12):
                continue

            if not self._TestBondWithNeighbor_(StartAtom, EndAtom, strict=params.StrictNeighborAtomRule):
                continue

            # [2]: Check bond structure (ring-bond)
            if not MolEngine._TestBondWithRingType_(bond, params=params):
                continue

            # [Optional]: Check isotope ?
            if params.UseIsotope:
                if StartAtom.GetIsotope() or EndAtom.GetIsotope():
                    print(StartAtom.GetIsotope(), EndAtom.GetIsotope())
                if StartAtom.GetSymbol() == 'H' and IsNotHydrogen(StartAtom):
                    pass
                elif EndAtom.GetSymbol() == 'H' or IsNotHydrogen(EndAtom):
                    pass
                else:
                    continue

            # [4]: Break bond
            new_mol, need_kekulize, error = MolEngine.BreakBond(mol, StartAtom.GetIdx(), EndAtom.GetIdx())
            if error:
                warning(f'Found invalid bond at index {bond.GetIdx()} >> REMOVED.')
                continue

            # Convert the 2 molecules into a SMILES string
            bondType = ComputeBondType(bond)
            Smi2A, Smi2B = MolEngine.CastBrokenMolIntoRadicals(new_mol, isomericSmiles=isIsomeric)
            if Smi2A == 'NONE' and Smi2B == 'NONE':
                warning(f'Found invalid bond at index {bond.GetIdx()} >> REMOVED.')
                continue

            reactions.append([smiles, CanonSmiles, IsoSmiles, Smi2A, Smi2B, bond.GetIdx(), bondType])

        self._cache.clear()
        TestState(len(reactions) != 0, 'The bond option or the molecule (SMILES) is NOT valid.')
        if params.ZeroDuplicate:
            reactions = MolEngine.RemoveDuplicate(reactions)
        # reactions.sort(key=lambda value: int(value[5]))  # Don't need but left here if any future issue
        if params.AddReverseReaction:
            reactions = MolEngine._MakeDuplicate_(reactions)
        return mol, reactions

    @staticmethod
    def EmbedBondNote(mol: Union[Mol, RWMol], mapping: Dict[int, str], inplace: bool = False) -> RWMol:
        RemovedHydroList: List[int] = []
        COPIED_MOL: Union[Mol, RWMol] = mol if inplace else RWMol(mol)
        for bond in PyGetBonds(COPIED_MOL):
            bIdx: int = bond.GetIdx()
            try:
                bond.SetProp('bondNote', mapping[bIdx])
            except (ValueError, IndexError, KeyError):
                # Remove if that bond is hydrogen-based, ignore if it is connected to aromatic ring but
                # not the carbon
                beginAtom = bond.GetBeginAtom()
                endAtom = bond.GetEndAtom()

                if not IsNotHydrogen(beginAtom):
                    if not (endAtom.GetIsAromatic() and endAtom.GetAtomicNum() != 6):
                        RemovedHydroList.append(beginAtom.GetIdx())
                if not IsNotHydrogen(endAtom):
                    if not (beginAtom.GetIsAromatic() and beginAtom.GetAtomicNum() != 6):
                        RemovedHydroList.append(endAtom.GetIdx())

        # Clean excessive hydrogen
        RemovedHydroList.sort(reverse=True)
        for atom in RemovedHydroList:
            COPIED_MOL.RemoveAtom(atom)
        Sanitize(COPIED_MOL)
        return COPIED_MOL

    @staticmethod
    def Display(reactions: List, ignoreRadicalWhenDisplay: str = None):
        TestState(len(reactions) != 0,
                  'The bond selection or the molecule (SMILES) is not valid. Please check again.')
        TestState(len(reactions[0]) == len(_LABEL), 'Invalid Output.')

        SMILES: str = reactions[0][1]
        print(f'Input Smiles: {reactions[0][0]}')
        print(f'Canonical Smiles: {reactions[0][1]}  <--->  Isomeric Smiles: {reactions[0][2]}')
        for reaction in reactions:
            if reaction[1] != SMILES:
                print(f'Input Smiles: {reaction[0]}')
                print(f'Canonical Smiles: {reaction[1]}  <--->  Isomeric Smiles: {reaction[2]}')
                SMILES = reaction[1]
            print('-' * 40)
            print(f'Bond Index: {reaction[5]} <--> Bond Type: {reaction[6]}')
            if not (ignoreRadicalWhenDisplay and reaction[3] == ignoreRadicalWhenDisplay):
                print(f'Radical: {reaction[3]}')
            if not (ignoreRadicalWhenDisplay and reaction[4] == ignoreRadicalWhenDisplay):
                print(f'Radical: {reaction[4]}')
        print('-' * 40)

    @staticmethod
    def Export(reactions: List, filename: str) -> pd.DataFrame:
        df = pd.DataFrame(data=reactions, columns=_LABEL, index=None)
        ExportFile(DataFrame=df, FilePath=filename)
        return df

    def GetBondOnMols(self, smiles: List[str], params: Optional[BondParams] = None) -> List[List]:
        reactions: List[List] = []
        smiles_list: List[str] = []
        if params is None:
            params = BondParams()

        for smi in smiles:
            result = self.GetBond(smiles=smi, params=params)
            smiles_list.append(result[0])
            reactions.extend(result[1])
        return reactions

    def GetBondInGroup(self, template: str, groups: Tuple[List[str], ...],
                       params: Optional[BondParams] = None) -> List[List]:
        matching = _RE_Compiler.findall(string=template)
        InputFullCheck(groups, name='groups', dtype='Tuple')
        TestState(len(set(matching)) == len(groups),
                  'The SMILES template is not compatible with the arg::groups.')
        iterable = groups
        if len(groups) != 1:
            from itertools import product
            iterable = product(*groups)  # Maintain duplication
        if params is None:
            params = BondParams()
        mols: List[str] = [template.format(combination) if not isinstance(groups, tuple) else
                           template.format(*combination) for combination in iterable]
        return self.GetBondOnMols(mols, params=params)

    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _IsIsomericSmiles_(*SMILES: str) -> bool:
        return any(string.find('/') != -1 or string.find('@') != -1 for string in SMILES)

    @staticmethod
    def _OptMol_(smiles: str, params: BondParams) -> Tuple[RWMol, Tuple[str, str], bool]:
        mol: Mol = SmilesToSanitizedMol(smiles, addHs=False)
        CanonicalSmiles: str = SmilesFromMol(mol, isomericSmiles=False)
        IsomericSmiles: str = SmilesFromMol(mol, isomericSmiles=True)

        TestStateByWarning(IsCanonString(smiles, mode='SMILES', useChiral=True),
                           f' This SMILES={smiles} is not canonical by RDKit notion.')

        iso = MolEngine._IsIsomericSmiles_(smiles, CanonicalSmiles, IsomericSmiles)
        if not params.UseIsotope:
            m = SmilesToSanitizedMol(IsomericSmiles if params.UseCanonicalOutput else smiles, addHs=True)
        else:
            m = mol
        return RWMol(m), (CanonicalSmiles, IsomericSmiles), iso

    def _BuildAtomBondType_(self, CoreAtomBondType: Optional[Union[str, Tuple[str, str]]]) -> bool:
        TestState(CoreAtomBondType is None or len(list(CoreAtomBondType)) <= 2,
                  msg='The :arg:`CoreAtomBondType` should be a string or a tuple of two-strings or None.')
        if CoreAtomBondType is None:
            self._cache['AtomBond'] = None
            return True

        temp = list(CoreAtomBondType)
        if len(temp) == 0:
            self._cache['AtomBond'] = None
            return True

        for atom in temp:
            try:
                _: int = _PERIODIC_TABLE.GetAtomicNumber(atom)
            except RuntimeError:
                warning(f'CoreAtomBondType is invalid: {CoreAtomBondType}.')
                return False

        if len(temp) == 1:
            self._cache['AtomBond'] = temp[0]
        else:
            temp.sort(reverse=False)
            self._cache['AtomBond'] = f'{temp[0]}-{temp[1]}'

        return True

    def _TestAtomBondType_(self, StartAtom, EndAtom) -> bool:
        AcceptBond = self._cache.get('AtomBond', None)
        if AcceptBond is None:
            return True
        if StartAtom.GetSymbol() == AcceptBond or EndAtom.GetSymbol() == AcceptBond:
            return True
        if '{}-{}'.format(*sorted([StartAtom.GetSymbol(), EndAtom.GetSymbol()])) == AcceptBond:
            return True
        return False

    def _BuildNeighborAtomMap_(self, NeighborAtoms: Optional[List[str]]) -> None:
        self._cache['neighbor'] = NeighborAtoms

    def _TestBondWithNeighbor_(self, StartAtom, EndAtom, strict: bool = False) -> bool:
        if self._cache['neighbor'] is None or not self._cache['neighbor']:
            return True

        StartIdx: int = StartAtom.GetIdx()
        EndIdx: int = EndAtom.GetIdx()

        ThisNeighbors: List[str] = [atom.GetSymbol() for atom in StartAtom.GetNeighbors() if atom.GetIdx() != EndIdx]
        for atom in EndAtom.GetNeighbors():
            if atom.GetIdx() != StartIdx:
                ThisNeighbors.append(atom.GetSymbol())
        HashedNeighbor = set(ThisNeighbors)
        SavedNeighbors = self._cache['neighbor']
        for atom in set(SavedNeighbors):
            if strict and (atom not in HashedNeighbor or ThisNeighbors.count(atom) != SavedNeighbors.count(atom)):
                return False
            elif not strict and (ThisNeighbors.count(atom) < SavedNeighbors.count(atom)):
                return False
        return True

    @staticmethod
    def _TestBondWithRingType_(bond, params: BondParams) -> bool:
        STAT = (params.AromaticRing, params.NonAromaticRing, params.NonAromaticRingAttached,
                params.AromaticRingAttached, params.NonRing)
        if not any(STAT):
            return False
        if all(STAT):
            return True

        result = DetermineBondState(bond)
        return (params.AromaticRing and result['aro-ring']) or \
            (params.NonAromaticRing and result['non-aro-ring']) or \
            (params.NonRing and result['non-ring']) or \
            (params.NonAromaticRingAttached and result['non-aro-ring-att']) or \
            (params.AromaticRingAttached and result['aro-ring-att'])

    @staticmethod
    def IsKekulizeWhenBreakBond(mol: Mol, *args: int) -> bool:
        if len(args) == 0 or len(args) > 2:
            raise ValueError('The number of arguments should be 1 or 2.')
        if len(args) == 1:
            bond = mol.GetBondWithIdx(args[0])
        else:
            bond = mol.GetBondBetweenAtoms(args[0], args[1])
        return bond.GetIsAromatic() or bond.GetBeginAtom().GetIsAromatic() or bond.GetEndAtom().GetIsAromatic()

    @staticmethod
    def BreakBond(mol: Mol, StartAtomIdx: int, EndAtomIdx: int) -> Tuple[RWMol, bool, bool]:
        need_kekulize = MolEngine.IsKekulizeWhenBreakBond(mol, StartAtomIdx, EndAtomIdx)
        SanitizeError = False

        rw_mol: RWMol = RWMol(mol)
        if need_kekulize:
            Kekulize(rw_mol, clearAromaticFlags=True)
        rw_mol.RemoveBond(StartAtomIdx, EndAtomIdx)
        rw_mol.GetAtomWithIdx(StartAtomIdx).SetNoImplicit(True)
        rw_mol.GetAtomWithIdx(EndAtomIdx).SetNoImplicit(True)

        try:
            Sanitize(rw_mol)
        except Exception as e:
            print(f'Sanitize Error: {e}')
            SanitizeError = True

        return rw_mol, need_kekulize, SanitizeError

    @staticmethod
    def CastBrokenMolIntoRadicals(mol: Mol, isomericSmiles: bool = True):
        try:
            Smi1A, Smi1B = sorted(MolToSmiles(mol).split('.'))

            temp = MolFromSmiles(Smi1A)
            if temp is None:
                return 'NONE', 'NONE'
            Smi2A = MolToSmiles(RemoveHs(MolFromSmiles(Smi1A)), isomericSmiles=isomericSmiles)

            temp = MolFromSmiles(Smi1B)
            if temp is None:
                return 'NONE', 'NONE'
            Smi2B = MolToSmiles(RemoveHs(MolFromSmiles(Smi1B)), isomericSmiles=isomericSmiles)
        except ValueError:  # Ring-bond Found
            try:
                Smi2A: str = MolToSmiles(RemoveHs(mol), isomericSmiles=isomericSmiles)
                Smi2B = 'NONE'
            except ValueError:  # Un-usual path >> Should not be called
                Smi2A, Smi2B = 'NONE', 'NONE'
        return Smi2A, Smi2B

    @staticmethod
    def RemoveDuplicate(reactions: List[List], radical_index: Tuple[int, int] = (3, 4)) -> List[List]:
        N: int = len(reactions)
        state: List[bool] = [False] * N
        for i in range(0, N):
            if state[i]:
                continue
            smi1, smi2 = reactions[i][radical_index[0]], reactions[i][radical_index[1]]
            if smi1 == 'NONE' or smi2 == 'NONE':
                continue
            for j in range(i + 1, N):
                if not state[j]:
                    if smi1 == reactions[j][radical_index[0]] and smi2 == reactions[j][radical_index[1]]:
                        state[j] = True
                    elif smi1 == reactions[j][radical_index[1]] and smi2 == reactions[j][radical_index[0]]:
                        state[j] = True
        return [reactions[idx] for idx in range(N) if not state[idx]]

    @staticmethod
    def _MakeDuplicate_(reactions: List[List]) -> List[List]:
        result: List[List] = []
        for reaction in reactions:
            smi1, smi2 = reaction[3], reaction[4]
            result.append(reaction)
            if smi1 != smi2:
                temp = reaction.copy()
                temp[3] = smi2
                temp[4] = smi1
                result.append(temp)
        return result
