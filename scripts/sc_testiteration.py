import numpy as np
from rdkit import Chem
from time import perf_counter
import gc
from Bit2Edge.molUtils.molUtils import PyCacheEdgeConnectivity, PyGetAtoms, PyGetBondWithIdx


def mean_and_std(arr: list) -> tuple:
    arr = np.array(arr)
    return np.mean(arr), np.std(arr)

if __name__ == '__main__':
    RUNS: int = 1000
    mol = Chem.MolFromSmiles('CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O')
    gc.disable()

    # [1]: Test GetAtoms()
    timing_array = []
    for _ in range(RUNS):
        start = perf_counter()
        for a in mol.GetAtoms():
            pass
        timing_array.append(perf_counter() - start)
    print(f'GetAtoms(): {mean_and_std(timing_array)}')

    # [2]: Test GetAtomWithIdx()
    timing_array = []
    for _ in range(RUNS):
        start = perf_counter()
        for i in range(mol.GetNumAtoms()):
            mol.GetAtomWithIdx(i)
        timing_array.append(perf_counter() - start)
    print(f'GetAtomWithIdx(): {mean_and_std(timing_array)}')

    # [3]: Test GetAtomsWithCache
    timing_array = []
    PyCacheEdgeConnectivity(mol)
    for _ in range(RUNS):
        start = perf_counter()
        for a in PyGetAtoms(mol):
            pass
        timing_array.append(perf_counter() - start)
    print(f'GetAtomsWithCache(): {mean_and_std(timing_array)}')

    # Test GetBondWithIdx()
    RUNS: int = 10000
    timing_array = []
    idx: int = 0
    for _ in range(RUNS):
        start = perf_counter()
        mol.GetBondWithIdx(idx)
        timing_array.append(perf_counter() - start)
    print(f'GetBondWithIdx(): {mean_and_std(timing_array)}')

    # Test GetBondWithCache()
    timing_array = []
    idx: int = 0
    for _ in range(RUNS):
        start = perf_counter()
        PyGetBondWithIdx(mol, idx)
        timing_array.append(perf_counter() - start)
    print(f'GetBondWithIdx(): {mean_and_std(timing_array)}')