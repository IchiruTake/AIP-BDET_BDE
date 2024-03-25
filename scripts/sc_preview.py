from typing import Dict, List, Tuple
from rdkit.Chem import Mol
from Bit2Edge.molUtils.molUtils import PyGetAtomWithIdx, PyGetBondNeighborsFilter


def FindEnvOfRadiusN(mol: Mol, radius, nbrStack: List[Tuple], bondMap: Dict[int, int], useHs: bool,
                     atomMap: Dict[int, int] = None) -> Tuple[int, List[int], List[int]]:
    IsAtomMapNotNone: bool = atomMap is not None
    distances = [len(bondMap)]
    for dist in range(1, radius + 1, 1):
        if len(nbrStack) == 0:
            return dist - 1, [*bondMap], distances

        nextLayer = []
        for atomIdx, bondIdx, bond in nbrStack:
            if bondIdx not in bondMap:
                bondMap[bondIdx]: int = dist
                nextAtomIdx: int = bond.GetOtherAtomIdx(atomIdx)
                if IsAtomMapNotNone and nextAtomIdx not in atomMap:
                    atomMap[nextAtomIdx] = dist

                if dist < radius and len(mol.PyEdgeGraph[nextAtomIdx]) > 1:
                    nextAtom = PyGetAtomWithIdx(mol, nextAtomIdx)
                    for nextBond, nextBondIdx in PyGetBondNeighborsFilter(mol, nextAtom, mapping=bondMap):
                        if useHs or nextBond.GetOtherAtom(nextAtom).GetAtomicNum() != 1:
                            nextLayer.append((nextAtomIdx, nextBondIdx, nextBond))
        distances.append(len(bondMap))
        nbrStack = nextLayer

    return radius, [*bondMap], distances
