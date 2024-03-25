from typing import Union, Tuple

from rdkit.Chem.rdchem import Mol, Atom, RWMol, Bond


class PyCacheMol(object):

    __slots__ = ('_mol', '_atoms', '_bonds', '_id_tracker', 'PyEdgeGraph')

    def __init__(self, mol: Union[Mol, RWMol]):
        self._mol: Union[Mol, RWMol] = mol
        self._atoms: Tuple[Atom, ...] = tuple(mol.GetAtoms())
        self._bonds: Tuple[Bond, ...] = tuple(mol.GetBonds())
        self._id_tracker = {'atom': set(), 'bond': set()}
        for atom in self._atoms:
            self._id_tracker['atom'].add(id(atom))
        for bond in self._bonds:
            self._id_tracker['bond'].add(id(bond))
        self.PyEdgeGraph = None

    @property
    def mol(self) -> Union[Mol, RWMol]:
        return self._mol

    @property
    def atoms(self) -> Tuple[Atom, ...]:
        return self._atoms

    @property
    def bonds(self) -> Tuple[Bond, ...]:
        return self._bonds

    def GetAtom(self, idx: int) -> Atom:
        return self._atoms[idx]

    def GetBond(self, idx: int) -> Bond:
        return self._bonds[idx]

    def GetPyEdgeGraph(self, force: bool = False) -> Tuple[Tuple[Bond, ...], ...]:
        if not force and self.PyEdgeGraph is not None:
            return self.PyEdgeGraph

        self.PyEdgeGraph = [[] for _ in range(len(self._atoms))]
        for bond in self._bonds:
            # Get Index
            b_sidx: int = bond.GetBeginAtomIdx()
            b_eidx: int = bond.GetEndAtomIdx()
            # Fill AtomMap
            self.PyEdgeGraph[b_sidx].append(bond)
            self.PyEdgeGraph[b_eidx].append(bond)

        for idx, value in enumerate(self.PyEdgeGraph):
            self.PyEdgeGraph[idx]: Tuple[Bond, ...] = tuple(value)
        self.PyEdgeGraph: Tuple = tuple(self.PyEdgeGraph)
        return self.PyEdgeGraph

    def _CheckAtomIdx(self, atomIdx: int) -> None:
        if atomIdx < 0 or atomIdx >= len(self._atoms):
            raise ValueError(f'Atom index {atomIdx} is out of range.')

    def _CheckBondIdx(self, bondIdx: int) -> None:
        if bondIdx < 0 or bondIdx >= len(self._bonds):
            raise ValueError(f'Bond index {bondIdx} is out of range.')

    def _CheckAtom(self, atom: Atom) -> None:
        if id(atom) not in self._id_tracker['atom']:
            raise ValueError(f'Atom {atom} does not belong to this molecule.')

    def _CheckBond(self, bond: Bond) -> None:
        if id(bond) not in self._id_tracker['bond']:
            raise ValueError(f'Bond {bond} does not belong to this molecule.')

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def PyGetBondNeighborsWithIdx(self, atomIdx: int) -> Tuple[Bond, ...]:
        self._CheckAtomIdx(atomIdx)
        return self.GetPyEdgeGraph()[atomIdx]

    def PyGetBondNeighbors(self, atom: Atom) -> Tuple[Bond, ...]:
        self._CheckAtom(atom)
        return self.PyGetBondNeighborsWithIdx(atom.GetIdx())

    def PyGetBondNeighborsFilter(self, atomIdx: int, mapping: dict) -> Tuple[Tuple[Bond, int], ...]:
        result = []
        for bond in self.PyGetBondNeighborsWithIdx(self, atomIdx):
            bond_idx: int = bond.GetIdx()
            if bond_idx not in mapping:
                result.append((bond, bond_idx))
        return tuple(result)

    def IsNotObsoleteAtomWithIdx(self, atomIdx: int) -> bool:
        connectivity = self.GetPyEdgeGraph()[atomIdx]
        return len(connectivity) > 1

    def IsNotObsoleteAtom(self, atom: Atom) -> bool:
        self._CheckAtom(atom)
        return self.IsNotObsoleteAtomWithIdx(atom.GetIdx())

    def PyGetAtomNeighborsWithIdx(self, atomIdx: int) -> Tuple[Atom, ...]:
        self._CheckAtomIdx(atomIdx)
        connectivity = self.GetPyEdgeGraph()[atomIdx]
        result = []
        for bond in connectivity:
            result.append(self._atoms[bond.GetOtherAtomIdx(atomIdx)])
        return tuple(result)

    def PyGetAtomNeighbors(self, atom: Atom) -> Tuple[Atom, ...]:
        self._CheckAtom(atom)
        return self.PyGetAtomNeighborsWithIdx(atom.GetIdx())
