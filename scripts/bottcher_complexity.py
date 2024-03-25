from __future__ import print_function

from math import log, log2
import os
import sys
from rdkit import Chem
from rdkit.Chem import RDConfig

from Bit2Edge.utils.helper import GetIndexOnArrangedData

sys.path.append(os.path.join(RDConfig.RDContribDir,'ChiralPairs'))
import ChiralDescriptors
from Bit2Edge.molUtils.molUtils import SmilesToSanitizedMol, PyCacheEdgeConnectivity
import pandas as pd
import numpy as np
from Bit2Edge.dataObject.FileParseParams import FileParseParams

def ComputeMol(smiles: str, hs: bool = False) -> Chem.Mol:
    mol = SmilesToSanitizedMol(smiles, addHs=hs)
    PyCacheEdgeConnectivity(mol)
    mol.dist_matrix = Chem.GetDistanceMatrix(mol)
    return mol
# D
#
# Current failures: Does not distinguish between cyclopentyl and pentyl (etc.)
#                   and so unfairly underestimates complexity.
def GetChemicalNonequivs(atom, mol):
    substituents = [[], [], [], []]
    result = ChiralDescriptors.determineAtomSubstituents(atom.GetIdx(), mol, mol.dist_matrix)[0]
    for item, key in enumerate(result):
        for subatom in result[key]:
            substituents[item].append(mol.GetAtomWithIdx(subatom).GetSymbol())
            # Logic to determine e.g. whether repeats of CCCCC are cyclopentyl and pentyl or two of either

    num_unique_substituents = len(set(tuple(tuple(substituent) for substituent in substituents if substituent)))
    return num_unique_substituents


# E
#
# The number of different non-hydrogen elements or isotopes (including deuterium
# and tritium) in the atom's microenvironment.
#
# CH4 - the carbon has e_i of 1
# Carbonyl carbon of an amide e.g. CC(=O)N e_i = 3
#     while N and O have e_i = 2
#
def GetBottcherLocalDiversity(atom, mol):
    neighbors = []
    for neighbor_bond in mol.PyEdgeGraph[atom.GetIdx()]:
        neighbor = neighbor_bond.GetOtherAtom(atom)
        neighbors.append(neighbor.GetAtomicNum())

    return len(set(neighbors)) + int(atom.GetAtomicNum() not in neighbors)


# S
#
# RDKit marks atoms where there is potential for isomerization with a tag
# called _CIPCode. If it exists for an atom, note that S = 2, otherwise 1.
def GetNumIsomericPossibilities(atom):
    try:
        if atom.GetProp('_CIPCode'):
            return 2
    except KeyError:
        pass
    return 1

# V
#
# The number of valence electrons the atom would have if it were unbonded and
# neutral
# Maybe able to use RDKit's GetExplicitValence() function
VALENCE = {
    'H': 1, 'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1, 'Fr': 1,    # Alkali Metals
    'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2, 'Ra': 2,           # Alkali Earth Metals
    # Transition Metals???
    'B': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3, 'Nh': 3,
    'C': 4, 'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4, 'Fl': 4,
    'N': 5, 'P': 5, 'As': 5, 'Sb': 5, 'Bi': 5, 'Mc': 5,             # Pnictogens
    'O': 6, 'S': 6, 'Se': 6, 'Te': 6, 'Po': 6, 'Lv': 6,             # Chalcogens
    'F': 7, 'Cl': 7, 'Br': 7, 'I': 7, 'At': 7, 'Ts': 7,             # Halogens
    'He': 8, 'Ne': 8, 'Ar': 8, 'Kr': 8, 'Xe': 8, 'Rn': 8, 'Og': 8   # Noble Gases
}
def GetNumValenceElectrons(atom):
    # Maybe able to use RDKit's GetExplicitValence() function
    return VALENCE.get(atom.GetSymbol(), atom.GetTotalValence())


# B
#
# Represents the total number of bonds to other atoms with V_i*b_i > 1, so
# basically bonds to atoms other than Hydrogen
#
# Here we can leverage the fact that RDKit does not even report Hydrogens by
# default to simply loop over the bonds. We will have to account for molecules
# that have hydrogens turned on before we can submit this code as a patch
# though.
#
# TODO: Create a dictionary for atom-B value pairs for use when AROMATIC is detected in bonds.
def GetBottcherBondIndex(atom, mol):
    b_sub_i_ranking = 0
    bonds = []
    for bond in mol.PyEdgeGraph[atom.GetIdx()]:
        if bond.GetBeginAtom().GetAtomicNum() == 1 or bond.GetEndAtom().GetAtomicNum() == 1:
            continue
        bonds.append(str(bond.GetBondType()))
    for bond in bonds:
        if bond == 'SINGLE':
            b_sub_i_ranking += 1
        elif bond == 'DOUBLE':
            b_sub_i_ranking += 2
        elif bond == 'TRIPLE':
            b_sub_i_ranking += 3
    if 'AROMATIC' in bonds:
        # This list can be expanded as errors arise.
        if atom.GetAtomicNum() == 6:
            b_sub_i_ranking += 3
        elif atom.GetAtomicNum() == 7:
            b_sub_i_ranking += 2
    return b_sub_i_ranking

def GetBottcherComplexity(smiles: str, hs: bool = False, debug=False):
    complexity = 0
    mol = ComputeMol(smiles, hs)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)

    atom_stereo_classes = []
    atoms_corrected_for_symmetry = []
    for atom in mol.PyAtoms:
        if atom.GetAtomicNum() == 1:
            continue
        if atom.GetProp('_CIPRank') in atom_stereo_classes:
            continue

        atoms_corrected_for_symmetry.append(atom)
        atom_stereo_classes.append(atom.GetProp('_CIPRank'))

    for atom in atoms_corrected_for_symmetry:
        d = GetChemicalNonequivs(atom, mol)
        e = GetBottcherLocalDiversity(atom, mol)
        s = GetNumIsomericPossibilities(atom)
        V = GetNumValenceElectrons(atom)
        b = GetBottcherBondIndex(atom, mol)
        try:
            complexity += d * e * s * log2(V * b)
        except ValueError:
            # print(d, e, s, V, b)
            pass
        if debug:
            print(str(atom.GetSymbol()))
            print('\tSymmetry Class: ' + str(atom.GetProp('_CIPRank')))
            print('\tNeighbors: ')
            print('\tBonds: ')
            print('\tCurrent Parameter Values:')
            print('\t\td_sub_i: ', d)
            print('\t\te_sub_i: ', e)
            print('\t\ts_sub_i: ', s)
            print('\t\tV_sub_i: ', V)
            print('\t\tb_sub_i: ', b)
    return complexity

if __name__ == '__main__':
    caffeine = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    print(f'Current Complexity Score of {caffeine}: {GetBottcherComplexity(caffeine)}')

    caffeine = 'C'
    print(f'Current Complexity Score of {caffeine}: {GetBottcherComplexity(caffeine, hs=False)}')

    caffeine = 'C'
    print(f'Current Complexity Score of {caffeine}: {GetBottcherComplexity(caffeine, hs=True)}')

    caffeine = 'CC'
    print(f'Current Complexity Score of {caffeine}: {GetBottcherComplexity(caffeine, hs=False)}')

    caffeine = 'CC'
    print(f'Current Complexity Score of {caffeine}: {GetBottcherComplexity(caffeine, hs=True)}')

    p = FileParseParams(mol=0, radical=(1, 2), bIdx=3, bType=4, target=5)
    df = pd.read_csv('../model/test_benchmark/Testing Case #5 - No Outlier.csv', header=0)
    result = GetIndexOnArrangedData(df.values, cols=p.Mol(), keys=str, get_last=False)

    complexities = []
    for row, smiles in result:
        # print(f'Current Complexity Score of {smiles}: ', end='')
        complexity = GetBottcherComplexity(smiles, hs=True)
        complexities.append(complexity)
        # print(complexity)
    print('Average Complexity Score: ', np.mean(complexities))
    print('Standard Deviation of Complexity Score: ', np.std(complexities))
    print('Max Complexity Score: ', np.max(complexities))
    print('Min Complexity Score: ', np.min(complexities))
    print(f'P99 Complexity Score: {np.percentile(complexities, 0.5)} - {np.percentile(complexities, 99.5)}')
    print(f'P95 Complexity Score: {np.percentile(complexities, 2.5)} - {np.percentile(complexities, 97.5)}')

