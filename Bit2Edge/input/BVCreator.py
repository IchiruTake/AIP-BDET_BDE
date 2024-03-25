# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from typing import Callable, List, Optional, Tuple, Union

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import AssignStereochemistry

from Bit2Edge.input.Fingerprint.Daylight import RDK, LayRDK, Pattern, Avalon, MACCS
from Bit2Edge.input.Fingerprint.FpUtils import BVAbstractEngine
from Bit2Edge.input.Fingerprint.Morgan import ECFP, FCFP, MHECFP, MHFCFP, SECFP
from Bit2Edge.input.Fingerprint.PathBased import AtomPairs, Torsion
from Bit2Edge.utils.verify import TestState
from Bit2Edge.molUtils.molUtils import SmilesToSanitizedMol


class BVCreator:
    """
    This class is an internal worker managed by BVManager which is responsible to convert one 
    molecule into a series of bit-vector. However, there are no API caller in this class, as we
    want to fully optimize string concatenation/joining. It is also a base class to support
    parallelism in BVManager.
    """

    __slots__ = ('name', 'isREWorker', 'engines', 'functions', '_labels',
                 '_Hs', '_none')

    def __init__(self, index: int, name: str, isREWorker: bool):
        TestState(len(name) == 1, 'The name should have one letter only.')
        self.name: str = name
        self.isREWorker: bool = isREWorker

        # Capture only usable BVCreator
        self.engines: List[BVAbstractEngine] = BVCreator._CreatePipeline_(index, isREWorker, toFunction=False)
        self.functions: Tuple[Callable, ...] = tuple(engine.MolToBitVect for engine in self.engines)

        self._labels: Optional[List[str]] = None
        self.GetLabels(force=False)

        # Caching-Level 1 --> O(1)
        self._Hs: str = self._CacheHydrogen_()
        self._none: str = '0' * self.GetFpVecSize()

    # --------------------------------------------------------------------------------------------------------
    def GetLabels(self, force: bool = False, copy: bool = True):
        if not force and self._labels is not None:
            return self._labels if not copy else list(self._labels)

        # engine.GetFpStringTemplate() is a constant string
        TEMPLATES = [engine.GetFpStringTemplate() for engine in self.engines]
        result = [f'{self.name}:{template}{bitId}' for template, engine in zip(TEMPLATES, self.engines)
                  for bitId in range(0, engine.GetFpBits())]

        if self._labels is None or force:
            self._labels = tuple(result)
        return result

    @staticmethod
    def _CreatePipeline_(index: int, isREWorker: bool, toFunction: bool = False) \
            -> List[Union[Callable, BVAbstractEngine]]:
        engines: Tuple[BVAbstractEngine, ...] = \
            (
                ECFP(index, isREWorker), FCFP(index, isREWorker), MHECFP(index, isREWorker),
                MHFCFP(index, isREWorker), SECFP(index, isREWorker), RDK(index, isREWorker),
                LayRDK(index, isREWorker), AtomPairs(index, isREWorker), Torsion(index, isREWorker),
                Avalon(index, isREWorker), Pattern(index, isREWorker), MACCS(index, isREWorker)
            )

        if not toFunction:
            return [engine for engine in engines if engine.GetFpStatus()]
        return [engine.MolToBitVect for engine in engines if engine.GetFpStatus()]

    def __eq__(self, other):
        if other is self:
            return True
        if not isinstance(other, BVCreator):
            raise TypeError(f'Incorrect comparison between {self.__class__} and {type(other)}.')

        # Run the quick evaluation
        if self.isREWorker != other.isREWorker or len(self.engines) != len(other.engines):
            return False

        return all(t_engine == o_engine for t_engine, o_engine in zip(self.engines, other.engines))

    # --------------------------------------------------------------------------------------------------------
    def GetFpVecSize(self) -> int:
        return len(self.GetLabels(force=False, copy=False))

    def __len__(self) -> int:
        return sum(len(engine) for engine in self.engines)

    def _CacheHydrogen_(self) -> str:
        mol: Mol = SmilesToSanitizedMol('[H]')
        AssignStereochemistry(mol)  # NULL EFFECT ON HYDROGEN ???
        return ''.join([func(mol).ToBitString() for func in self.functions])

    def GetHydroFpVect(self) -> str:
        return self._Hs

    def GetEmptyFpVect(self) -> str:
        return self._none

    # Main Function
    def GetFpBitVectAPI(self, mol: Optional[Mol], pool: List[str]) -> None:
        if mol is None:
            pool.append(self._none)
            return None

        if self.isREWorker:
            if mol.GetNumAtoms() == 1 and mol.GetAtomWithIdx(0).GetAtomicNum() == 1:
                pool.append(self._Hs)
                return None

        # [2]: Evaluate the molecule
        for func in self.functions:
            pool.append(func(mol).ToBitString())
        return None
