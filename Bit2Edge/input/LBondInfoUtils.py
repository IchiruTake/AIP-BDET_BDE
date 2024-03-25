# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module stored some utility functions for LBondInfo
# --------------------------------------------------------------------------------

from collections import defaultdict
from logging import warning
from typing import Dict, List, Optional, Tuple, Union

from rdkit.Chem import FindMolChiralCenters
from rdkit.Chem.rdCIPLabeler import AssignCIPLabels
from rdkit.Chem.rdchem import BondStereo, Mol, StereoSpecified, StereoType
from rdkit.Chem.rdmolfiles import MolToSmiles
from rdkit.Chem.rdmolops import (AssignStereochemistry, FindPotentialStereo,
                                 FindPotentialStereoBonds, RemoveHs)

from Bit2Edge.config.userConfig import DATA_FRAMEWORK as dFramework
from Bit2Edge.molUtils.molUtils import PyGetAtoms, PyGetBonds, PyGetAtomWithIdx, \
    PyGetBondWithIdx, IsNotObsoleteAtom, PyGetBondNeighborsFilter
from Bit2Edge.utils.verify import TestState

# ----------------------------------------------------------------------------
# Prepare labels
__LABEL_TAGS_DTYPE = Union[str, List[str]]


def _LabelCorrection_(value: Union[str, List[str], Tuple[str, ...]]) -> str:
    _ForceString_ = lambda string: string if isinstance(string, str) else str(string)
    if isinstance(value, (List, Tuple)):
        return '-'.join([_ForceString_(string) for string in value])
    return _ForceString_(value)


def _AddSingle_(FeatureList: List[__LABEL_TAGS_DTYPE], size: int,
                labels: List[str], tables: List, length: List) -> int:
    tables.append({})
    for idx, token in enumerate(FeatureList):
        labels.append(_LabelCorrection_(token))
        if isinstance(token, (List, Tuple)):
            for sub_token in token:
                tables[-1][sub_token] = size + idx
        else:
            tables[-1][token] = size + idx
    size += len(FeatureList)
    length.append(size)
    return size


def _AddMulti_(*FeatureLists: Tuple[List[__LABEL_TAGS_DTYPE], ...], size: int,
               labels: List[str], tables: List, lengths: List) -> int:
    """
        This method systematically unraveled all features in `FeatureLists` parameter and construct
        the associated `labels` around it from `FeatureLists` -> `FtList` -> `SubFtList` -> `Token`.
        If iterable is available inside `Token`, it would be merged before adding into `labels`.
    """
    count = len(FeatureLists[0])
    names = []
    for i, FtList in enumerate(FeatureLists):
        TestState(count == len(FtList), f'The size of each FeatureList must be equal (={count}) and compatible.')
        names.append(f'L{i} - ')

    last: int = len(tables)  # Getting the position of last update.
    for i, (name, FtList) in enumerate(zip(names, FeatureLists)):
        for j, SubFtList in enumerate(FtList):
            index: int = last + j

            if i == 0:  # Perform once only
                tables.append(defaultdict(lambda *args: []))
                lengths.append([])

            table: Union[defaultdict, dict] = tables[index]  # cache
            for idx, token in enumerate(SubFtList):
                labels.append(f'{name}{_LabelCorrection_(token)}')
                if isinstance(token, (List, Tuple)):
                    for sub_token in token:  # Used for atom type grouping
                        table[sub_token].append(size + idx)
                else:
                    table[token].append(size + idx)

            # Resolve missing values.
            for token, value in table.items():
                if len(value) != i + 1:
                    table[token].append(None)

            size += len(SubFtList)
            lengths[index].append(size)
    return size


# ---------------------------------------------------------------------------------------------------------
# Stereo-chemistry
# Global variables helps for fast indexing rather than create object in memory (reduce memory footprint)
def _LegacyFindMolChiralCenters_(mol, includeUnassigned: bool = False) -> bool:
    NonDefinedAtom: int = 0
    for atom in PyGetAtoms(mol):
        if (atom.HasProp('_CIPCode') and atom.GetProp('_CIPCode') == '?') or \
                (includeUnassigned and atom.HasProp('_ChiralityPossible')):
            NonDefinedAtom += 1
            if NonDefinedAtom == 2:
                return False
    return True


def _OptFindMolChiralCenters_(mol, includeCIP: bool = True, includeUnassigned: bool = False) -> bool:
    NonDefinedAtom: int = 0
    StereoItems = FindPotentialStereo(mol)

    AtomTetrahedral = StereoType.Atom_Tetrahedral
    Specified = StereoSpecified.Specified

    if includeCIP:
        atomsToLabel = []
        bondsToLabel = []
        BondDouble = StereoType.Bond_Double
        for si in StereoItems:
            if si.type == AtomTetrahedral:
                atomsToLabel.append(si.centeredOn)
            elif si.type == BondDouble:
                bondsToLabel.append(si.centeredOn)
        AssignCIPLabels(mol, atomsToLabel=atomsToLabel, bondsToLabel=bondsToLabel)

    for si in StereoItems:
        if si.type == AtomTetrahedral and (includeUnassigned or si.specified == Specified):
            atm = PyGetAtomWithIdx(mol, si.centeredOn)
            if (includeCIP and (atm.HasProp('_CIPCode') and atm.GetProp('_CIPCode') == '?')) or \
                    (not includeCIP and (not si.specified or str(si.descriptor) == '?')):
                NonDefinedAtom += 1
                if NonDefinedAtom == 2:
                    return False
    return True


def SearchChiralCenters(mol: Mol, includeUnassigned: bool = False, includeCIP: bool = True,
                        useLegacyImplementation: bool = True) -> bool:
    if useLegacyImplementation:
        return _LegacyFindMolChiralCenters_(mol, includeUnassigned)
    return _OptFindMolChiralCenters_(mol, includeCIP, includeUnassigned)


def FindStereoAtomsBonds(mol: Mol, NeedAssignStereochemistry: bool = True) -> None:
    if NeedAssignStereochemistry:
        AssignStereochemistry(mol, force=False, cleanIt=True, flagPossibleStereoCenters=True)
    FindPotentialStereoBonds(mol, cleanIt=True)


def GetMolStereo(mol: Mol, useNone: bool = True) -> Tuple[Dict, Dict]:
    MolStereo = {}
    MolStereoReverse: Dict[str, int] = \
        {
            BondStereo.STEREOZ: 0, BondStereo.STEREOE: 0,
            BondStereo.STEREOCIS: 0, BondStereo.STEREOTRANS: 0,
            BondStereo.STEREOANY: 0, BondStereo.STEREONONE: 0,
        }

    for bond in PyGetBonds(mol):
        bondStereo: str = bond.GetStereo()
        if useNone or bondStereo == BondStereo.STEREONONE:
            continue
        bIdx: int = bond.GetIdx()
        MolStereo[bIdx] = bondStereo
        MolStereoReverse[bondStereo] += 1
    return MolStereo, MolStereoReverse


def GetStereoChemistry(mol: Mol, MolStereoR: Optional[Dict[str, int]] = None, applyLegacy: bool = True) -> int:
    """
        Determine the molecule stereo-chemistry: Check if the given molecule can build and calculate
        accurate enthalpies with the given stereo-chemistry information.
        Inspired by Peter C. St. John (2020)

        NOTE: We have done a bit of optimization ahead to prevent if-else checking to construct
        a consistent pipeline in which you can see it in method LBICreator.GetLBondInfoAPI()

        Arguments:
        ---------
        - mol (Mol): RDKit molecule
        - molStereoR (Dict): The second return value of function `GetMolStereo()`
        - applyLegacy (bool): Whether to use the legacy approach of 'AssignStereochemistry()'

        Returns:
        ---------
        - arg[0]: The stereochemistry counting state

    """
    if not SearchChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=applyLegacy):
        return 0

    if MolStereoR is None:
        return not any(bond.GetStereo() == BondStereo.STEREOANY for bond in PyGetBonds(mol))
    return MolStereoR[BondStereo.STEREOANY] == 0


def GetStereoChemistryLegacy(mol: Mol, applyLegacy: bool = True) -> Tuple[dict, bool]:
    if not applyLegacy:
        AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    FindPotentialStereoBonds(mol)

    centers = FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=applyLegacy)
    unassigned_atom = sum(1 for center in centers if center[1] == '?')

    # bondList: List[str] = []
    all_bonds: int = 0
    unassigned_bond: int = 0
    for bond in PyGetBonds(mol):
        value = bond.GetStereo()
        if value != BondStereo.STEREONONE:
            # bondList.append(value)
            all_bonds += 1
        if value == BondStereo.STEREOANY:
            unassigned_bond += 1

    series = {'atom': len(centers) - unassigned_atom, 'non_atom': unassigned_atom,
              'bond': all_bonds - unassigned_bond, 'non_bond': unassigned_bond}

    return series, unassigned_atom <= 1 and unassigned_bond == 0


def _WarnStereochemistry_(mol: Mol, previous_message: Optional[str]) -> Optional[str]:
    message: str = f' Molecule {MolToSmiles(RemoveHs(mol))} has undefined stereochemistry.'
    if previous_message is None or previous_message == message:
        warning(message)
        return message
    return previous_message


def CheckInvalidStereoChemistry(mol: Mol, applyLegacy: bool = True, previous_message: str = None) -> str:
    """
        This function is to check the validity of stereo-chemistry by John, et al (2020).
        Note that there is no specific purpose for this method.

        Arguments:
        ---------
            - mol: The RDKit molecule
            - previous_message (str): The string showing the previous message
            - applyLegacy (bool): Whether to use the legacy approach of 'AssignStereochemistry()'

        Returns:
        ---------
             - A string message that is either valid or not
    """
    # series = GetStereoChemistryLegacy(mol=RemoveHs(mol), applyLegacy=applyLegacy)[0]
    # if series["non_atom"] != 0 or series["non_bond"] != 0:
    if GetStereoChemistry(RemoveHs(mol), applyLegacy=applyLegacy) == 0:
        return _WarnStereochemistry_(mol, previous_message=previous_message)
    return previous_message


def GetRadicalsStereoChemistry(FragMolX: Mol, FragMolY: Mol, applyLegacy: bool = True) -> int:
    """
        Implementation of retrieving stereo-chemistry with radical

        Arguments:
        ---------
            - FragMolX: The RDKit molecule (usually the radical)
            - FragMolY: The RDKit molecule (usually the radical)

        Returns:
        ---------
            - A boolean-based integer of `GetStereoChemistry(FragMolX) or GetStereoChemistry(FragMolY)`
    """
    return GetStereoChemistry(FragMolX, None, applyLegacy) or GetStereoChemistry(FragMolY, None, applyLegacy)


# --------------------------------------------------------------------------------------------------
# Cis-Trans Identifier
def GetCisTransLabels() -> List[str]:
    labels: List[str] = ['Stereo Chemistry']  # E: Trans --- Z: Cis

    if dFramework['Cis-Trans Encoding']:
        labels.append('Count Z-Bonds')
        labels.append('Count E-Bonds')

    if dFramework['Cis-Trans Counting']:
        # ['Cis Bond', 'Cis Atom (C)', 'Cis Atom (N)', 'Trans Bond', 'Trans Atom (C)', 'Trans Atom (N)']
        for bondType in ('Cis', 'Trans'):
            labels.append(f'{bondType} Bond')
            for atom in dFramework['Cis-Trans Atoms']:
                labels.append(f'{bondType} Atom ({atom})')
    return labels


def DetectCisTrans(mol: Mol, bondIdx: int, result: List[int], molStereo: Dict[int, str],
                   useNone: bool = True, safe: bool = False) -> List[int]:
    # [0]: Generate Input/Output Array
    cistrans_atoms: Tuple[str, ...] = dFramework['Cis-Trans Atoms']
    blockSize: int = 0 - (len(cistrans_atoms) + 1)  # Cache for later speed up
    result.extend([0] * (blockSize * -2))

    # [1]: Check possibility of atoms if potentially possess cis-trans bond
    # Cis-Trans: Z =~= "Cis", "E" =~= "Trans"
    bond = PyGetBondWithIdx(mol, bondIdx)
    mapping = (bondIdx, )

    # [2]: Get all data needed (Cis - Trans Identifier)
    # Bond Database store the bond index which is part of cis-trans
    for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
        if safe or IsNotObsoleteAtom(mol, atom=atom):
            for neighborBond, nIdx in PyGetBondNeighborsFilter(mol, atom, mapping=mapping):
                # [2.1]: Get the neighbor index that is contained possibility of being a cis-trans bond
                # but not current prediction bond.
                if molStereo is None or nIdx not in molStereo:
                    continue
                B_STEREO: str = molStereo[nIdx]
                # This may be raise if applyNoneChecking = False (NONE_STATE=False)
                if B_STEREO == BondStereo.STEREOANY or (useNone and B_STEREO == BondStereo.STEREONONE):
                    continue

                # Cis-Z = First block, speed up achieved here
                ptr: int = blockSize * (1 + int(B_STEREO == BondStereo.STEREOZ or B_STEREO == BondStereo.STEREOCIS))
                result[ptr] += 1

                OTHER_SYMBOL = neighborBond.GetOtherAtom(atom).GetSymbol()
                if OTHER_SYMBOL in cistrans_atoms:
                    result[ptr + 1 + cistrans_atoms.index(OTHER_SYMBOL)] += 1
    return result


def CountCisTrans(MolStereoR: Dict[str, int]) -> Tuple[int, int]:
    return MolStereoR[BondStereo.STEREOZ] + MolStereoR[BondStereo.STEREOCIS], \
           MolStereoR[BondStereo.STEREOE] + MolStereoR[BondStereo.STEREOTRANS]
