# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module stored some utility functions for feature engineering (legacy).
# --------------------------------------------------------------------------------
from logging import info
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from rdkit.Chem.rdchem import Mol, RWMol
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import AddHs, RemoveHs, SanitizeMol

from Bit2Edge.dataObject.FileParseParams import FileParseParams
from Bit2Edge.utils.cleaning import DeleteArray, GetRemainingIndexToLimit
from Bit2Edge.utils.file_io import ExportFile, FixPath, ReadFile, RemoveExtension
from Bit2Edge.utils.helper import GetIndexOnArrangedData
from Bit2Edge.utils.verify import (InputCheckIterable, InputCheckRange,
                                   InputFastCheck, InputFullCheck,
                                   MeasureExecutionTime, TestState)

_MOL_TYPE = Union[Mol, RWMol]


# ----------------------------------------------------------------------------------------------------------------
# [1]: Utility function
def CheckEquivalent(FragMolX: _MOL_TYPE, FragMolY: _MOL_TYPE, FragArr1: List[_MOL_TYPE],
                    FragArr2: List[_MOL_TYPE], rematch: bool = True) -> bool:
    """
    This function performed molecular matching using the RDKit implementation. To validate whether
    the molecule is identical, set :arg:`rematch`=True. Note that the position in the list must for
    cross-referencing must be the same companion.

    Arguments:
    ---------

    FragMolX : Mol or RWMol
        The first RDKit molecule/fragment. Order can be swapped with :arg:`FragMolY`.
    
    FragMolY : Mol or RWMol
        The second RDKit molecule/fragment. Order can be swapped with :arg:`FragMolX`.
    
    FragArr1 : List[Mol or RWMol]
        A first list of molecules for comparison and matching. Order can be swapped with :arg:`FragArr2`.
    
    FragArr2 : List[Mol or RWMol]
        A second list of molecules for comparison and matching. Order can be swapped with :arg:`FragArr1`.
    
    rematch : bool
        If True, we performed molecular matching. Otherwise, it is just a substructure matching. Default to True.
    
    Returns:
    -------
    
    A boolean value indicating whether the two molecules are identical.

    """
    InputFullCheck(FragArr1, name='FragArr1', dtype='List-Tuple', delimiter='-')
    InputFullCheck(FragArr2, name='FragArr2', dtype='List-Tuple', delimiter='-')

    temp = (len(FragArr1), len(FragArr2))
    TestState(temp[0] == temp[1], f'Two lists are not totally equivalent with {temp[0]} vs {temp[1]}.')
    InputFullCheck(rematch, name='rematch', dtype='bool')

    # cacheState[0]: The index of matching
    # cacheState[1]: True if FragMolX ~ FragArr2 ; False if FragMolX ~ FragArr1
    # cacheState[2]: The state of matching
    result: bool = False
    cacheState: Tuple = (-1, None, False)  # the index, the
    for i, (frag_1, frag_2) in enumerate(zip(FragArr1, FragArr2)):
        # Opposite (Indirect) Comparison ---> Forward (Direct) Comparison
        # Fast Mode
        if FragMolX.HasSubstructMatch(frag_2, False, False, False) and \
                FragMolY.HasSubstructMatch(frag_1, False, False, False):
            result = True
            cacheState = (i, True, False)
            break

        if FragMolX.HasSubstructMatch(frag_1, False, False, False) and \
                FragMolY.HasSubstructMatch(frag_2, False, False, False):
            result = True
            cacheState = (i, False, False)
            break

        if FragMolX.HasSubstructMatch(frag_2, True, True, True) and \
                FragMolY.HasSubstructMatch(frag_1, True, True, True):
            result = True
            cacheState = (i, True, True)
            break

        if FragMolX.HasSubstructMatch(frag_1, True, True, True) and \
                FragMolY.HasSubstructMatch(frag_2, True, True, True):
            result = True
            cacheState = (i, False, True)
            break

    if not rematch:
        return result

    # To make a reverse check (molecule identity), we have to guarantee that the molecule combination
    # must be True to be forwardly executed, otherwise discard result
    if not result:
        return False

    # order cannot be None here, comboIndex cannot be -1.
    if cacheState[1]:
        molX = FragMolX
        molY = FragMolY
    else:
        molX = FragMolY
        molY = FragMolX
    SubstructMatchParams = [cacheState[2]] * 3
    if FragArr2[cacheState[0]].HasSubstructMatch(molX, *SubstructMatchParams):
        if FragArr1[cacheState[0]].HasSubstructMatch(molY, *SubstructMatchParams):
            return True

    # Support condition
    ReverseSubstructMatchParams = [not cacheState[2]] * 3
    if FragArr2[cacheState[0]].HasSubstructMatch(molX, *ReverseSubstructMatchParams):
        if FragArr1[cacheState[0]].HasSubstructMatch(molY, *ReverseSubstructMatchParams):
            return True

    info('Two fragment combinations may be not identical. Please check them.')
    p1 = (False, False, False)
    p2 = (True, True, True)
    for i, (frag_1, frag_2) in enumerate(zip(FragArr1, FragArr2)):
        if i == cacheState[0]:
            continue
        if frag_2.HasSubstructMatch(molX, *p1) and frag_1.HasSubstructMatch(molY, *p1):
            return True
        if frag_2.HasSubstructMatch(molX, *p2) and frag_1.HasSubstructMatch(molY, *p2):
            return True
    return False


def GetBondIndex(ParentMol: _MOL_TYPE, FragMolX: _MOL_TYPE, FragMolY: _MOL_TYPE,
                 current: int = 0, maxBonds: int = None, verbose: bool = False, rematch: bool = True) -> int:
    """
    This function will find the bond index that can break the :arg:`ParentMol` into two FragMols, which 
    are :arg:`FragMolX` and :arg:`FragMolY`. This function will perform checking from :arg:`current` to
    :arg:`maxBonds - 1`.

    Arguments:
    ---------

    ParentMol : Mol
        The molecule use for bond-breaking searching
    
    FragMolX : Mol
        The first RDKit molecule/fragment. Order can be swapped with :arg:`FragMolY`.
    
    FragMolY : Mol
        The second RDKit molecule/fragment. Order can be swapped with :arg:`FragMolX`.
    
    current : int
        The starting position to start searching. Default to be 0.
    
    maxBonds : int
        The maximum bond inside the molecule. Default to be None. If None, the function will use the
        number of bonds inside the molecule (ParentMol.GetNumBonds()).
    
    verbose : bool
        Whether to show up progress meters. Default to be False.
    
    rematch : bool
        Whether to make reversing validation. This argument is similar in function :meth:`CheckEquivalent()`
    
    Returns:
    -------

    An integer that is the bond index that can break the molecule into two fragments. If no bond can
    break the molecule, the function will return -1.

    """
    # Hyper-parameter Verification
    if True:
        FragX, FragY = MolToSmiles(RemoveHs(FragMolX)), MolToSmiles(RemoveHs(FragMolY))
        if maxBonds is None:
            maxBonds: int = ParentMol.GetNumBonds()
        else:
            InputCheckRange(maxBonds, name='maxBonds', maxValue=ParentMol.GetNumBonds(), minValue=current,
                            rightBound=False)
        InputCheckRange(current, name='current', maxValue=maxBonds, minValue=0)
        InputFullCheck(verbose, name='verbose', dtype='bool')
        InputFullCheck(rematch, name='rematch', dtype='bool')

    for idx in range(current, maxBonds):
        bond = ParentMol.GetBondWithIdx(idx)
        if bond.GetIsAromatic() or bond.IsInRing():
            continue

        TempMol = RWMol(ParentMol)
        BeginAtom, EndingAtom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        TempMol.RemoveBond(BeginAtom, EndingAtom)

        TempMol.GetAtomWithIdx(BeginAtom).SetNoImplicit(True)
        TempMol.GetAtomWithIdx(EndingAtom).SetNoImplicit(True)

        # Call SanitizeMol to update radicals (Used when kekulize before)
        SanitizeMol(TempMol)

        # Convert the 2 molecules into a SMILES string
        Smi1A, Smi1B = sorted(MolToSmiles(TempMol).split('.'))
        Mol1A, Mol1B = AddHs(MolFromSmiles(Smi1A)), AddHs(MolFromSmiles(Smi1B))

        Smi2A, Smi2B = MolToSmiles(MolFromSmiles(Smi1A), isomericSmiles=False), \
            MolToSmiles(MolFromSmiles(Smi1B), isomericSmiles=False)
        Mol2A, Mol2B = AddHs(MolFromSmiles(Smi2A)), AddHs(MolFromSmiles(Smi2B))

        Smi3A, Smi3B = MolToSmiles(MolFromSmiles(Smi1A), isomericSmiles=True), \
            MolToSmiles(MolFromSmiles(Smi1B), isomericSmiles=True)
        Mol3A, Mol3B = AddHs(MolFromSmiles(Smi3A)), AddHs(MolFromSmiles(Smi3B))

        Smi4A, Smi4B = MolToSmiles(RemoveHs(MolFromSmiles(Smi1A))), \
            MolToSmiles(RemoveHs(MolFromSmiles(Smi1B)))
        Mol4A, Mol4B = AddHs(MolFromSmiles(Smi4A)), AddHs(MolFromSmiles(Smi4B))

        if verbose:
            print('Current Bond Index:', idx)
            print(f'Smiles 1A: {Smi1A} - Smiles 1B: {Smi1B}')
            print(f'Smiles 2A: {Smi2A} - Smiles 2B: {Smi2B}')
            print(f'Smiles 3A: {Smi3A} - Smiles 3B: {Smi3B}')
            print(f'Smiles 4A: {Smi4A} - Smiles 4B: {Smi4B}')

        if CheckEquivalent(FragMolX=FragMolX, FragMolY=FragMolY, FragArr1=[Mol1A, Mol2A, Mol3A, Mol4A],
                           FragArr2=[Mol1B, Mol2B, Mol3B, Mol4B], rematch=rematch):
            return idx

        # Perform Smiles Checking if failed
        if FragX in (Smi1A, Smi2A, Smi3A, Smi4A) and FragY in (Smi1B, Smi2B, Smi3B, Smi4B):
            return idx
        if FragY in (Smi1A, Smi2A, Smi3A, Smi4A) and FragX in (Smi1B, Smi2B, Smi3B, Smi4B):
            return idx
    return -1


@MeasureExecutionTime
def ProfileDatabase(database: ndarray, params: FileParseParams) -> Dict[str, List[int]]:
    """
    This function will retrieve all the 'ring-status' bonds inside a given dataset.

    Arguments:
    ---------

    database : ndarray
        The database to be considered
    
    params : FileParseParams
        The parameters of the database, only the following parameters are used:
            - Mol()
            - BondIndex()
        
    Returns:
    -------

    A dictionary that contains the following keys:
        - 'Aro-Ring-Row': A list of integer of row for reactions of the aromatic-ring-attached bond.
        - 'Aro-Ring-Mol': A list of integer of (starting) row for molecule of the aromatic-ring-attached bond.
        - 'NonAro-Ring-Row': A list of integer of row for reactions of the non-aromatic-ring-attached bond.
        - 'NonAro-Ring-Mol': A list of integer of (starting) row for molecule of the non-aromatic-ring-attached bond.

        - 'Non-Ring-Row': A list of integer of row for reactions of the non-ring-attached bond.
        - 'Non-Ring-Mol': A list of integer of (starting) row for molecule of the non-ring-attached bond.
        - 'Ring-Row': A list of integer of row for reactions of the ring-membered bond.
        - 'Ring-Mol': A list of integer of (starting) row for molecule of the ring-membered bond.
    
    """
    if not InputFastCheck(database, 'ndarray'):
        database = np.asarray(database)
    InputCheckRange(params.Mol(), name='MoleculeCol', maxValue=database.shape[1], minValue=0)
    InputCheckRange(params.BondIndex(), name='BondIdxCol', maxValue=database.shape[1], minValue=0)

    # [1]: Prepare data
    IndexData: List = GetIndexOnArrangedData(database, cols=params.Mol(), get_last=True)
    BondIdxList: List = database[:, params.BondIndex()].tolist()
    status: Dict[str, List[int]] = \
        {
            'Aro-Ring-Row': [],
            'NonAro-Ring-Row': [],
            'Non-Ring-Row': [],
            'Ring-Row': [],

            'Aro-Ring-Mol': [],
            'NonAro-Ring-Mol': [],
            'Non-Ring-Mol': [],
            'Ring-Mol': []
        }

    def FillRows(key: str, MolRow: int, ReactionRow) -> None:
        if MolRow != status[f'{key}-Mol'][-1]:
            status[f'{key}-Mol'].append(MolRow)
        if ReactionRow != status[f'{key}-Row'][-1]:
            status[f'{key}-Row'].append(ReactionRow)
        return None

    # [2]: Looping by request
    for molSet in range(0, len(IndexData) - 1):
        begin, end = IndexData[molSet][0], IndexData[molSet + 1][0]
        SMILES: str = str(IndexData[molSet][1])
        molecule: Mol = AddHs(MolFromSmiles(SMILES))

        for row in range(begin, end):
            bond = molecule.GetBondWithIdx(int(BondIdxList[row]))
            startAtom, endAtom = bond.GetBeginAtom(), bond.GetEndAtom()
            startAtomInRing: bool = startAtom.IsInRing()
            endAtomInRing: bool = endAtom.IsInRing()

            if startAtomInRing or endAtomInRing:
                if startAtomInRing and endAtomInRing and bond.IsInRing():
                    FillRows(key='Ring', MolRow=begin, ReactionRow=row)

                if (startAtomInRing and startAtom.GetIsAromatic()) or (endAtomInRing and endAtom.GetIsAromatic()):
                    FillRows(key='Aro-Ring', MolRow=begin, ReactionRow=row)
                else:
                    FillRows(key='NonAro-Ring', MolRow=begin, ReactionRow=row)

            else:
                FillRows(key='Non-Ring', MolRow=begin, ReactionRow=row)

    # [3]: Get the result
    print('Result: ')
    for key, value in status.items():
        if 'Mol' in key:
            print(f'{key}: {len(value)} molecules ({round(100 * len(value) / len(IndexData), 2)} %).')
        else:
            print(f'{key}: {len(value)} BDEs ({round(100 * len(value) / database.shape[0], 2)} %).')

    return status


def ProfileDatabaseByFile(InputFileName: str, params: FileParseParams) -> Dict[str, List[int]]:
    array, columns = ReadFile(FilePath=InputFileName, header=0, get_values=True, get_columns=True)
    status = ProfileDatabase(database=array, params=params)

    for key, value in status.items():
        if 'Mol' in key:
            Keyword = key.replace('-Mol', '')
            FileName = f"{RemoveExtension(FixPath(InputFileName, 'csv'), '.csv')} - {Keyword}.csv"
            df = pd.DataFrame(data=array[value, :], columns=columns)
            ExportFile(DataFrame=df, FilePath=FileName)

    return status


# ----------------------------------------------------------------------------------------------------------------
# [2]: Internal function applied in Creator.FeatureEngineer
def _ConstraintDataset_(data: Union[pd.DataFrame, ndarray], limitTest: Union[int, float, List, Tuple] = None) \
        -> Union[pd.DataFrame, ndarray]:
    """
    This function cutout, constraint, and define the data points in the :arg:`data` argument by 
    the :arg:`limitTest` argument.

    Arguments:
    ---------

    data : A pandas.DataFrame or numpy.ndarray. The dataset we want to extract.

    limitTest : Integer, float, list or tuple. 
        If integer, it defines the number of samples on top of the dataset. If float and between 
        (0, 1), it defines the percentage of number of samples on top of the dataset. If a list,
        it selects the samples defined, if a tuple, it selects the samples in a slice. If None,
        select the whole dataset.
    
    Returns:
    -------

    data: A new dataset would be created if the manipulation occured, otherwise return a view

    """
    if limitTest is None:
        return data

    name: str = 'limitTest'
    InputFullCheck(limitTest, name=name, dtype='int-float-List-Tuple-None', delimiter='-', fastCheck=True)
    if InputFastCheck(limitTest, dtype='float'):
        InputCheckRange(limitTest, name=name, maxValue=1, allowFloatInput=True, rightBound=True)
        if limitTest == 1 or limitTest == 1.0:
            limitTest: int = data.shape[0] if limitTest == 1 or limitTest == 1.0 else int(limitTest * data.shape[0])

    if InputFastCheck(limitTest, dtype='int') and limitTest == data.shape[0]:
        return data

    if InputFastCheck(limitTest, dtype='int-Tuple', delimiter='-'):
        InputCheckRange(limitTest, name=name, maxValue=data.shape[0], minValue=0, leftBound=False,
                        rightBound=False)
        if isinstance(limitTest, Tuple):
            InputCheckIterable(value=limitTest, name=name, maxValue=data.shape[0], maxInputInside=2)
            if min(limitTest) == 0:
                limitTest = max(limitTest)

        if isinstance(limitTest, int):
            deleteLine: List[int] = list(range(limitTest, data.shape[0]))
        else:
            InputCheckIterable(value=limitTest, name=name, maxValue=data.shape[0])
            deleteLine: List[int] = list(range(0, min(limitTest)))
            deleteLine.extend(range(max(limitTest), data.shape[0]))
    else:
        limitTest.sort()
        deleteLine: List[int] = GetRemainingIndexToLimit(PrunedMask=limitTest, maxValue=data.shape[0])

    if isinstance(data, pd.DataFrame):
        return data.drop(labels=deleteLine, inplace=False)
    return DeleteArray(data, obj=deleteLine, axis=0)
