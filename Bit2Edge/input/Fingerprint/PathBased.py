# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect as AtomPair
from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprintAsBitVect as TTorsion
from rdkit.Chem.rdchem import Mol

from Bit2Edge.config.userConfig import DATA_FRAMEWORK as dFramework
from Bit2Edge.input.Fingerprint.FpUtils import BVAbstractEngine, _GetSpecialValue_


class AtomPairs(BVAbstractEngine):
    # __slots__ = ('_nBits', '_dist', '_chiral', '_seed', '_cache')

    def __init__(self, index: int, isREWorker: bool):
        nBits: int = _GetSpecialValue_(dFramework['AtomPairs_nBits'], index, isREWorker)
        chiral: bool = _GetSpecialValue_(dFramework['AtomPairs_Chiral'], index, isREWorker)
        # distance: int = 4       # nBitsPerEntry = 4
        super(AtomPairs, self).__init__(nBits, 4, chiral)

    def GetFpStringTemplate(self) -> str:
        return f'AP #'

    def MolToBitVect(self, mol: Mol):
        # http://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html
        return AtomPair(mol, nBits=self._nBits, includeChirality=self._chiral)


class Torsion(BVAbstractEngine):
    # __slots__ = ('_nBits', '_dist', '_chiral', '_seed', '_cache')

    def __init__(self, index: int, isREWorker: bool):
        nBits: int = _GetSpecialValue_(dFramework['Torsion_nBits'], index, isREWorker)
        distance: int = _GetSpecialValue_(dFramework['Torsion_Size'], index, isREWorker)
        chiral: bool = _GetSpecialValue_(dFramework['Torsion_Chiral'], index, isREWorker)
        super(Torsion, self).__init__(nBits, distance, chiral)

    def GetFpStringTemplate(self) -> str:
        return f'TT {self.GetFpDist()}/#'

    def MolToBitVect(self, mol: Mol):
        return TTorsion(mol, targetSize=self._dist, nBits=self._nBits, includeChirality=self._chiral)
