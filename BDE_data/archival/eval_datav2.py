import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import (Mol, RWMol)
from rdkit.Chem import Draw, rdDepictor, Kekulize
from rdkit.Chem.Draw.rdMolDraw2D import PrepareAndDrawMolecule

from Bit2Edge.dataObject.FileParseParams import FileParseParams
from Bit2Edge.input.MolProcessor.MolEngine import MolEngine
from Bit2Edge.utils.file_io import ReadFile, ExportFile, FixPath, RemoveExtension
from Bit2Edge.utils.helper import GetIndexOnArrangedData
from pprint import pprint, pformat
from time import perf_counter
from Bit2Edge.molUtils.molUtils import SmilesFromMol, SmilesToSanitizedMol, CopyAndKekulize, Sanitize, Kekulize
from rdkit.Chem import MolFromSmiles, MolToSmiles

def cleanup(src_file: str, dst_file: str, params: FileParseParams, ) -> None:
    radical = params.Radical()
    df = ReadFile(src_file, header=0)
    print(df.head(10))

    print(f"Start sorting over {df.shape[0]} rows.")
    sort_cols = [df.columns[params.Mol()], df.columns[params.BondIndex()]]
    df = df.sort_values(by=sort_cols, ignore_index=True)
    df_data = df.values.tolist()
    print(df.head(10))

    print(f"Start indexing over {df.shape[0]} rows.")
    df_data.sort(key=lambda x: (x[params.Mol()], x[params.BondIndex()]))
    index = GetIndexOnArrangedData(df_data, cols=params.Mol(), get_last=True, keys=None)
    print(f"Index: {pformat(index[:10])}")

    # Now we do the rows-filtering
    masks = [False] * index[-1][0]
    rows = []
    non_rows = []
    BLOCK_SIZE: int = 3000
    print(f"Start filtering over {index[-1][0]} rows with {len(index) - 1} partitions.")
    for i in range(0, len(index) - 1):
        start, mol = index[i]
        end = index[i + 1][0]
        partition: list = df_data[start:end]
        if i % BLOCK_SIZE == 0 and i != 0:
            print(f'Completed {i} partitions with progress {i / (len(index) - 1) * 100:.2f}%.')
            # print(f'Partition {i}: {partition}')

        for j in range(start, end):
            if masks[j] is True:
                continue
            rows.append(j)
            for k in range(j + 1, end):
                if masks[k] is True:
                    continue
                if partition[k - start][radical[0]] == partition[j - start][radical[0]] and \
                        partition[k - start][radical[1]] == partition[j - start][radical[1]]:
                    masks[k] = True
                    non_rows.append(k)
                elif partition[k - start][radical[0]] == partition[j - start][radical[1]] and \
                        partition[k - start][radical[1]] == partition[j - start][radical[0]]:
                    masks[k] = True
                    non_rows.append(k)

    non_rows.sort()
    print(f'Rows: {len(rows)} rows --> Non-rows: {len(non_rows)} rows.')
    hashed_rows = set(rows)
    filtered = [row for row in range(0, index[-1][0]) if row not in hashed_rows]
    print(f'Is filter works correct: {len(filtered) == len(non_rows)}')
    if len(filtered) != len(non_rows):
        print("The filter is not working correctly.")
        exit(0)

    print(f"Start exporting from {src_file} to {dst_file}.")
    data_rows = [df_data[row] for row in rows]
    output = pd.DataFrame(data_rows, columns=df.columns, index=None)
    print(output.head(10))
    ExportFile(output, dst_file, index=False)
    ExportFile(pd.DataFrame(index, columns=list(range(2))),
               dst_file.replace('.csv', '_index.csv'), index=False)


def get_bondtype(src_file: str, dst_file: str, params: FileParseParams) -> None:
    radical = params.Radical()
    df = ReadFile(src_file, header=0)
    df_data = df.values.tolist()
    df_data.sort(key=lambda x: (x[params.Mol()], x[params.BondIndex()]))

    print(f"Start indexing over {df.shape[0]} rows.")
    index = GetIndexOnArrangedData(df_data, cols=params.Mol(), get_last=True, keys=None)
    print(f"Index: {pformat(index[:10])}")

    # Now we do the validation
    BLOCK_SIZE: int = 1000
    addition_data = {
        'bType': [],
        'radical1': [],
        'radical2': [],
        'radical1_nochiral': [],
        'radical2_nochiral': [],
        'fragment1_nochiral': [],
        'fragment2_nochiral': [],
    }
    t = perf_counter()
    print(f"Start filtering over {index[-1][0]} rows with {len(index) - 1} partitions.")
    for i in range(0, len(index) - 1):
        start, mol = index[i]
        end = index[i + 1][0]
        partition: list = df_data[start:end]
        m = SmilesToSanitizedMol(mol, addHs=True)
        if i % BLOCK_SIZE == 0 and i != 0:
            print(f'Completed {i} partitions under {perf_counter() - t:.2f} seconds '
                  f'with progress {i / (len(index) - 1) * 100:.2f}%.')
            t = perf_counter()
            pass

        for j in range(start, end):
            data_block = partition[j - start]
            bIdx: int = data_block[params.BondIndex()]
            bond = m.GetBondWithIdx(bIdx)
            start_atom = bond.GetBeginAtom().GetSymbol()
            end_atom = bond.GetEndAtom().GetSymbol()
            bType: str = '{}-{}'.format(*sorted([start_atom, end_atom]))

            # Validation
            Smi1_src, Smi2_src = data_block[radical[0]], data_block[radical[1]]
            Smi1_src_nochiral = MolToSmiles(MolFromSmiles(Smi1_src), isomericSmiles=False)
            Smi2_src_nochiral = MolToSmiles(MolFromSmiles(Smi2_src), isomericSmiles=False)
            addition_data['fragment1_nochiral'].append(Smi1_src_nochiral)
            addition_data['fragment2_nochiral'].append(Smi2_src_nochiral)
            try:
                smol, need_kekulize, err = MolEngine.BreakBond(m, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                Smi1, Smi2 = sorted(MolToSmiles(smol).split('.'))
                x1, x2 = MolFromSmiles(Smi1), MolFromSmiles(Smi2)
                Smi1, Smi1_nochiral = MolToSmiles(x1), MolToSmiles(x1, isomericSmiles=False)
                Smi2, Smi2_nochiral = MolToSmiles(x2), MolToSmiles(x2, isomericSmiles=False)
                addition_data['radical1'].append(Smi1)
                addition_data['radical2'].append(Smi2)
                addition_data['radical1_nochiral'].append(Smi1_nochiral)
                addition_data['radical2_nochiral'].append(Smi2_nochiral)

                if Smi1 == Smi1_src and Smi2 == Smi2_src:
                    addition_data['bType'].append(bType)
                    continue
                if Smi2 == Smi2_src and Smi1 == Smi1_src:
                    addition_data['bType'].append(bType)
                    continue
                print(f'Expected: ({Smi1_src}, {Smi2_src}) --> Result: ({Smi1}, {Smi2})')
                print(f'Identical validation failed on {partition[j - start]} at row {j + 2} in Excel.')

                if Smi1_nochiral == Smi1_src_nochiral and Smi2_nochiral == Smi2_src_nochiral:
                    addition_data['bType'].append(bType)
                    continue
                if Smi1_nochiral == Smi2_src_nochiral and Smi2_nochiral == Smi2_src_nochiral:
                    addition_data['bType'].append(bType)
                    continue

                addition_data['bType'].append('ERROR')
                print(f'Expected: ({Smi1_src}, {Smi2_src}) --> Result: ({Smi1}, {Smi2})')
                print(f'Chiral validation failed on {partition[j - start]} at row {j + 2} in Excel.')

            except Exception as e:
                print(f'Bond Type: {bType}')
                print('Is error sanitizing: {}'.format(err))
                print('Result: ({}, {})'.format(Smi1, Smi2))
                print(f'Expected: ({Smi1_src}, {Smi2_src})')
                print(f'Validation failed on {partition[j - start]} at row {j + 2} in Excel.')

                # Draw the molecule
                tool = rdMolDraw2D.MolDraw2DCairo(500, 500)
                draw_params = tool.drawOptions()
                draw_params.centreMoleculesBeforeDrawing = True
                draw_params.dummyIsotopeLabels = False
                draw_params.includeRadicals = False
                draw_params.clearBackground = True
                draw_params.dummiesAreAttachments = False
                draw_params.padding = 0.05
                draw_params.addBondIndices = True
                tool.DrawMolecule(m)
                tool.FinishDrawing()
                plt.show()
                tool.WriteDrawingText(f'failed_mol_{j + 2:06d}.png')
                raise e
            pass

    print(f"Start exporting from {src_file} to {dst_file}.")
    for key, value in addition_data.items():
        df[key] = value
    print(df.head(10))
    ExportFile(df, dst_file, index=False)
    ExportFile(pd.DataFrame(index, columns=list(range(2))),
               dst_file.replace('.csv', '_index.csv'), index=False)


def compare_duplicateremoval_between_python_and_excel(src_file: str):
    df = ReadFile(src_file, header=0)
    python_molecules = df['python_molecule'].values.tolist()
    excel_molecules = df['excel_molecule'].values.tolist()
    print(f'Python: {len(python_molecules)} molecules.')
    print(f'Excel: {len(excel_molecules)} molecules.')

    set_python = set(python_molecules)
    set_excel = set(excel_molecules)
    print(f'Python: {len(set_python)} molecules.')
    print(f'Excel: {len(set_excel)} molecules.')

    identical = set_python.intersection(set_excel)
    print(f'Identical: {len(identical)} molecules.')

    python_only = set_python.difference(set_excel)
    print(f'Python only: {len(python_only)} molecules.')
    print(f'Python-only molecules: {python_only}')

    excel_only = set_excel.difference(set_python)
    print(f'Excel only: {len(excel_only)} molecules.')
    print(f'Excel-only molecules: {excel_only}')

if __name__ == '__main__':
    # src = '../BDE_data/20211201_bonds_for_neighbors.csv'
    # dst = '../BDE_data/source_dataset_v2_test.csv'
    # p = FileParseParams(mol=1, bIdx=2, radical=(3, 4), bType=5, target=6)
    # cleanup(src, dst, p)

    # src = '../BDE_data/source_dataset_v2_test.csv'
    # dst = '../BDE_data/source_dataset_v2_test_bType.csv'
    # p = FileParseParams(mol=1, bIdx=2, radical=(3, 4), bType=5, target=6)
    # get_bondtype(src, dst, p)

    # smi = 'Brc1[nH]nc2ncncc12'
    # bIdx = 11
    # smol = checkmol(smi, bIdx)
    # Smi1, Smi2 = sorted(MolToSmiles(smol).split('.'))
    # print(Smi1, Smi2)
    # print(MolToSmiles(MolFromSmiles(Smi1)))
    # print(MolToSmiles(MolFromSmiles(Smi2)))

    src = '../BDE_data/source_dataset_v2_test_index.csv'
    compare_duplicateremoval_between_python_and_excel(src)

