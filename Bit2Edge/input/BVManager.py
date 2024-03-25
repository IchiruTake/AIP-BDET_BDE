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

from Bit2Edge.dataObject.InputState import InputState
from Bit2Edge.input.BVCreator import BVCreator
from Bit2Edge.utils.verify import TestState


def _GetFeatureLocation_(b_workers: Tuple[BVCreator, ...], r_workers: Tuple[BVCreator, ...]) \
        -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    # [1.1]: Define the size of bit vector based on workers
    EndLocation = [worker.GetFpVecSize() for worker in b_workers]
    if len(r_workers) != 0:
        EndLocation.extend([worker.GetFpVecSize() for worker in r_workers])

    # [1.2]: Define the ending point (slice) of each bit vect
    for i in range(1, len(EndLocation), 1):
        EndLocation[i] += EndLocation[i - 1]
    fpVectSize: int = sum(worker.GetFpVecSize() for worker in b_workers) + \
                      sum(worker.GetFpVecSize() for worker in r_workers)
    TestState(EndLocation[-1] == fpVectSize, 'Incorrect algorithm.')

    # [2]: Calculate the starting point (slice)
    StartLocation: List[int] = [0]
    for i in range(0, len(EndLocation) - 1, 1):
        StartLocation.append(EndLocation[i])

    TestState(len(StartLocation) == len(EndLocation), 'Incorrect algorithm.')
    return tuple(StartLocation), tuple(EndLocation)


def GetFeatureLocation(b_workers: Tuple[BVCreator, ...], r_workers: Tuple[BVCreator, ...],
                       mergeOption: bool = False):
    StartLocation, EndLocation = _GetFeatureLocation_(b_workers=b_workers, r_workers=r_workers)
    if not mergeOption:
        return StartLocation, EndLocation
    return tuple(zip(StartLocation, EndLocation))


def _TestRWorkers_(r_workers: Tuple[BVCreator, BVCreator]) -> None:
    if len(r_workers) == 0:
        return None

    TestState(len(r_workers) == 2, 'The number of r_workers should only be zero or two.')
    msg: str = 'Two radical environments do not have equivalent workers'
    R1, R2 = r_workers
    TestState(R1.GetFpVecSize() == R2.GetFpVecSize(), f'{msg}: Fingerprint Size.')
    for i, (r1_label, r2_label) in enumerate(zip(R1.GetLabels(), R2.GetLabels())):
        TestState(r1_label.split(':')[1] == r2_label.split(':')[1],
                  f'{msg}: Fingerprint Labels at Index={i} is different.')
    return None


__MOL__ = Optional[Mol]


class BVManager:
    """
    This class is a higher-level wrapper of BVCreator which controlled the behaviour
    of each BVCreator (worker). This class also supported parallelism as similar as

    Arguments:
    ---------

    mols : Iterable[Mol]
        A sequence of RDKit molecule
    
    Returns:
    -------

    A bit-vector represented as a bit string
    """

    __slots__ = ('b_workers', 'r_workers', '_WorkerFuncs', '_stereochemistry',
                 'GetBitVectAPI',)

    def __init__(self, stereochemistry: bool = True):
        # [1]: Make worker
        names: Tuple[str, ...] = InputState.GetNames()
        radius = InputState.GetRadius()

        # b_workers: Bond Environment Worker
        # r_workers: Radical Environment Worker
        self.b_workers: Tuple[BVCreator, ...] = \
            tuple(BVCreator(index=idx, name=name, isREWorker=False)
                  for idx, (name, r) in enumerate(zip(names, radius)))
        self.r_workers: Tuple[BVCreator, BVCreator] = tuple()

        self._stereochemistry: bool = stereochemistry
        # [2]: Collected Attribute
        # [2.1]: Accumulate all workers
        workers: List[BVCreator] = [worker for workers in (self.b_workers, self.r_workers) for worker in workers]
        # ._workers: List[BVCreator] = [worker for workers in (self.b_workers, self.r_workers) for worker in workers]
        self._WorkerFuncs: Tuple[Callable, ...] = tuple(worker.GetFpBitVectAPI for worker in workers)

        # Routing traffic to avoid for-loop ?
        mapping = {2: self._TwoMols_, 3: self._ThreeMols_, 4: self._FourMols_}
        self.GetBitVectAPI: Callable = mapping.get(len(workers), self._AnyMols_)

    def GetLabels(self) -> List[str]:
        res = self.b_workers[0].GetLabels(force=False, copy=True)
        for i in range(1, len(self.b_workers)):
            res.extend(self.b_workers[i].GetLabels(force=False, copy=False))
        for worker in self.r_workers:
            res.extend(worker.GetLabels(force=False, copy=False))
        return res

    def GetFeatureLocation(self, mergeOption: bool = False):
        return GetFeatureLocation(b_workers=self.b_workers, r_workers=self.r_workers, mergeOption=mergeOption)

    def __eq__(self, other):
        if other is self:
            return True
        if not isinstance(other, BVManager):
            raise TypeError(f'Incorrect comparison between {self.__class__} and {type(other)}.')

        if len(self.b_workers) != len(other.b_workers) or len(self.r_workers) != len(other.r_workers):
            return False

        if any(t_worker != o_worker for t_worker, o_worker in zip(self.b_workers, other.b_workers)):
            return False

        if any(t_worker != o_worker for t_worker, o_worker in zip(self.r_workers, other.r_workers)):
            return False
        return True

    def __len__(self) -> int:
        return sum(len(worker) for worker in self.b_workers) + sum(len(worker) for worker in self.r_workers)

    def GetWorkerCodes(self) -> Tuple[Callable, ...]:
        return self._WorkerFuncs

    def GetStereochemistry(self) -> bool:
        return self._stereochemistry

    # --------------------------------------------------------------------------------------------------------
    # Routing Function
    @staticmethod
    def _OpCode_(mol: __MOL__, function: Callable, pool: List[str]) -> None:
        # Hot-code
        if mol is not None:
            AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
        function(mol, pool)

    def _TwoMols_(self, mols: Union[List[__MOL__], Tuple[__MOL__, ...]]) -> str:
        pool: List[str] = []
        m1, m2 = mols
        f1, f2 = self._WorkerFuncs
        if self._stereochemistry:
            BVManager._OpCode_(m1, f1, pool)
            BVManager._OpCode_(m2, f2, pool)
        else:
            f1(m1, pool)
            f2(m2, pool)
        return ''.join(pool)

    def _ThreeMols_(self, mols: Union[List[__MOL__], Tuple[__MOL__, ...]]) -> str:
        pool: List[str] = []
        m1, m2, m3 = mols
        f1, f2, f3 = self._WorkerFuncs
        if self._stereochemistry:
            BVManager._OpCode_(m1, f1, pool)
            BVManager._OpCode_(m2, f2, pool)
            BVManager._OpCode_(m3, f3, pool)
        else:
            f1(m1, pool)
            f2(m2, pool)
            f3(m3, pool)
        return ''.join(pool)

    def _FourMols_(self, mols: Union[List[__MOL__], Tuple[__MOL__, ...]]) -> str:
        pool: List[str] = []
        m1, m2, m3, m4 = mols
        f1, f2, f3, f4 = self._WorkerFuncs
        if self._stereochemistry:
            BVManager._OpCode_(m1, f1, pool)
            BVManager._OpCode_(m2, f2, pool)
            BVManager._OpCode_(m3, f3, pool)
            BVManager._OpCode_(m4, f4, pool)
        else:
            f1(m1, pool)
            f2(m2, pool)
            f3(m3, pool)
            f4(m4, pool)
        return ''.join(pool)

    def _AnyMols_(self, mols: Union[List[__MOL__], Tuple[__MOL__, ...]]) -> str:
        pool: List[str] = []
        for mol, func in zip(mols, self._WorkerFuncs):
            if self._stereochemistry and mol is not None:
                AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
            func(mol, pool)
        return ''.join(pool)
