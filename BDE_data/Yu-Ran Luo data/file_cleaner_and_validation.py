from typing import List, Tuple

from rdkit.Chem import (AddHs, MolFromSmiles, MolToSmiles, RemoveHs, SanitizeMol, Kekulize)
from rdkit.Chem.rdchem import RWMol

import pandas as pd
from Bit2Edge.utils.helper import ReadFile, ExportFile, GetIndexOnArrangedData
from Bit2Edge.input.CreatorUtils import CheckEquivalent
from Bit2Edge.input.MolProcessor.MolEngine import MolEngine

def CanonSmiles(smi: str, useChiral: bool = True) -> str:
    return MolToSmiles(MolFromSmiles(smi), isomericSmiles=useChiral)


def Validate(path: str, output_path: str):
    df = ReadFile(path, header=0)
    Lines = {'mol': 0, 'radical': (1, 2), 'b_index': 3}
    df_shape: Tuple[int, int] = df.shape
    data = df.values.tolist()

    arr = [0] * df_shape[0]
    col = ['Canonical', 'IsCanonMol', 'IsSameMol', 'IsCorrectBond']

    for row, row_data in enumerate(data):
        res: List = [0] * len(col)

        smi: str = row_data[Lines['mol']]
        canon_smi: str = CanonSmiles(smi, True)
        res[0] = canon_smi
        res[1] = int(canon_smi == smi)
        if not res[1]:
            print('Row:', row + 2, 'Smi:', smi, 'Canon_Smi:', canon_smi)

            m1, m2 = MolFromSmiles(smi), MolFromSmiles(canon_smi)
            res[2] = int(m1.HasSubstructMatch(m2) and m2.HasSubstructMatch(m1))
            if not res[2]:
                print('Row:', row + 2, 'Two molecule is not the same ???')
        else:
            res[2] = 1

        mol = AddHs(MolFromSmiles(smi))

        bIndex = int(row_data[Lines['b_index']])

        rw_mol: RWMol = RWMol(mol)
        bond = mol.GetBondWithIdx(bIndex)
        rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        rw_mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).SetNoImplicit(True)
        rw_mol.GetAtomWithIdx(bond.GetEndAtomIdx()).SetNoImplicit(True)

        try:
            SanitizeMol(rw_mol)
        except Exception:
            try:
                rw_mol: RWMol = RWMol(mol)
                Kekulize(rw_mol, clearAromaticFlags=True)
                rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                rw_mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).SetNoImplicit(True)
                rw_mol.GetAtomWithIdx(bond.GetEndAtomIdx()).SetNoImplicit(True)
                SanitizeMol(rw_mol)
            except Exception as e:
                raise e

        Smi1A, Smi1B = sorted(MolToSmiles(rw_mol).split('.'))
        Smi2A = MolToSmiles(RemoveHs(MolFromSmiles(Smi1A)))
        Smi2B = MolToSmiles(RemoveHs(MolFromSmiles(Smi1B)))

        IsCorrectBond = 0
        radicals_str = row_data[Lines['radical'][0]], row_data[Lines['radical'][1]]
        if Smi1A == radicals_str[0] and Smi1B == radicals_str[1]:
            IsCorrectBond = 1
        elif Smi1A == radicals_str[1] and Smi1B == radicals_str[0]:
            IsCorrectBond = 1
        elif Smi2A == radicals_str[0] and Smi2B == radicals_str[1]:
            IsCorrectBond = 1
        elif Smi2A == radicals_str[1] and Smi2B == radicals_str[0]:
            IsCorrectBond = 1

        if not IsCorrectBond:
            FragArr1 = [MolFromSmiles(Smi1A), MolFromSmiles(Smi2A)]
            FragArr2 = [MolFromSmiles(Smi1B), MolFromSmiles(Smi2B)]
            mol_1 = MolFromSmiles(radicals_str[0])
            mol_2 = MolFromSmiles(radicals_str[1])
            try:
                if CheckEquivalent(FragMolX=mol_1, FragMolY=mol_2, FragArr1=FragArr1, FragArr2=FragArr2):
                    IsCorrectBond = 1
                    df.iloc[row, Lines['radical'][0]] = Smi2A
                    df.iloc[row, Lines['radical'][1]] = Smi2B
                else:
                    IsCorrectBond = -99
                    print('Row:', row + 2, 'Current:', (radicals_str[0], radicals_str[1]), 'Canon:', (Smi2A, Smi2B))
            except AttributeError:
                IsCorrectBond = -999
                print('Row:', row + 2, 'Current:', (radicals_str[0], radicals_str[1]), 'Canon:', (Smi2A, Smi2B))
                pass

        res[3] = IsCorrectBond
        arr[row] = res

    df2 = pd.DataFrame(data=arr, columns=col, index=None)

    new_df = pd.concat((df2, df), axis=1)
    return ExportFile(DataFrame=new_df, FilePath=output_path)


def RuleEmbedding(path: str, output_path: str):
    df = ReadFile(path, header=0)
    LINES = {'mol': 0, 'b_index': 3, 'bde': 5, 'std': 6, 'recommend': 9,
             'index': 11, 'valid': 17}
    COLS = {key: df.columns[value] for key, value in LINES.items()}
    df = df[df[COLS['valid']] != 0]
    df.sort_values([COLS['mol'], COLS['b_index']], inplace=True)

    data: List[List] = df.values.tolist()

    index_data = GetIndexOnArrangedData(array=df.values, cols=(LINES['mol'], LINES['b_index']), get_last=True)[1]
    array = [0] * (len(index_data) - 1)

    for combination in range(0, len(index_data) - 1, 1):
        start, end = index_data[combination][0], index_data[combination + 1][0]
        count: int = end - start

        rules = [int(data[row][LINES['recommend']]) for row in range(start, end)]
        bdes = [float(data[row][LINES['bde']]) for row in range(start, end)]
        stds = [float(data[row][LINES['std']]) for row in range(start, end)]

        # Rule Injection here
        result: List = data[start].copy()
        indexs = [row for row, rule in enumerate(rules) if rule != 0]
        nRecs: int = len(indexs)

        remainer = [row for row, rule in enumerate(rules) if rule == 0]
        result[LINES['recommend']] = len(indexs) * (-1)

        rec_bde = [bdes[i] for i in indexs]
        rec_std = [stds[i] for i in indexs]
        rem_bde = [bdes[i] for i in remainer]
        rem_std = [stds[i] for i in remainer]

        if len(indexs) == 0 or count == nRecs:
            result[LINES['bde']] = sum(bdes) / len(bdes)
            result[LINES['std']] = sum(stds) / len(stds)
        elif nRecs <= 2:
            if count == nRecs + 1:
                factor: float = 0.8
            elif count == nRecs + 2:
                factor: float = 0.7
            elif count == nRecs + 3:
                factor: float = 0.6
            elif count >= nRecs + 4:
                factor: float = 0.5
            else:
                raise ValueError()

            bde, std = 0, 0
            for rec_bde_i, rec_std_i in zip(rec_bde, rec_std):
                bde += (rec_bde_i * factor / len(rec_bde))
                std += (rec_std_i * factor / len(rec_std))
            for rem_bde_i, rem_std_i in zip(rem_bde, rem_std):
                bde += (rem_bde_i * (1 - factor) / len(rem_bde))
                std += (rem_std_i * (1 - factor) / len(rem_std))
            result[LINES['bde']] = bde
            result[LINES['std']] = std
        else:
            raise ValueError()
        array[combination] = result

    array.sort(key=lambda value: int(value[LINES['index']]))
    df2 = pd.DataFrame(data=array, columns=df.columns, index=None)
    df2[COLS['bde']] = df2[COLS['bde']].round(4)
    df2[COLS['std']] = df2[COLS['std']].round(3)
    ExportFile(DataFrame=df2, FilePath=output_path)


def DropDuplicateRadical(path: str, output_path: str):
    df = ReadFile(path, header=0)
    data: List[List] = df.values.tolist()
    index_data = GetIndexOnArrangedData(array=df.values, cols=0, get_last=True)
    result = []
    for combination in range(0, len(index_data) - 1, 1):
        start, end = index_data[combination][0], index_data[combination + 1][0]
        temp = data[start:end]
        r = MolEngine.RemoveDuplicate(temp, (1, 2))
        result.extend(r)

    df2 = pd.DataFrame(data=result, columns=df.columns, index=None)
    ExportFile(DataFrame=df2, FilePath=output_path)

def GetMols(path: str):
    df = ReadFile(path, header=0)
    index_data = GetIndexOnArrangedData(array=df.values, cols=0, get_last=True)
    arr = [index_data[i][1] for i in range(0, len(index_data) - 1)]
    return arr


if __name__ == '__main__':
    path = "../model/test_data/bondnet_CH_bond_v3.csv"
    out = "../model/test_data/bondnet_CH_bond_v3_cleaned.csv"
    # Validate(path=path, output_path=out)
    #
    # out = "../model/test_data/bondnet_CH_bond_v3_free.csv"
    # DropDuplicateRadical(path, out)
    mols = GetMols(path)
    idx = [33, 41, 42, 49, 63, 64, 71, 73, 81, 83, 90, 91, 96]
    for i in idx:
        print(i, mols[i])

    # RuleEmbedding(path='data/Exp-BDEs Dataset - V1 - (Clean Constant Distance).csv',
    #               output_path='data/[Cleaned x2] Exp-BDEs Dataset - V1.csv')
