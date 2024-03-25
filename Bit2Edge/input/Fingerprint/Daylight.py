# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import LayeredFingerprint, PatternFingerprint, RDKFingerprint

from Bit2Edge.config.userConfig import DATA_FRAMEWORK as dFramework
from Bit2Edge.input.Fingerprint.FpUtils import BVAbstractEngine, _GetSpecialValue_


class RDK(BVAbstractEngine):
    # __slots__ = ('_nBits', '_dist', '_chiral', '_seed', '_cache')

    def __init__(self, index: int, isREWorker: bool):
        nBits: int = _GetSpecialValue_(dFramework['RDKit_nBits'], index, isREWorker)
        distance: bool = _GetSpecialValue_(dFramework['RDKit_maxPath'], index, isREWorker)
        super(RDK, self).__init__(nBits, distance)

    def GetFpStringTemplate(self) -> str:
        return f'RDK {self.GetFpDist()}/#'

    def MolToBitVect(self, mol: Mol):
        return RDKFingerprint(mol, maxPath=self._dist, fpSize=self._nBits)


class LayRDK(BVAbstractEngine):
    # __slots__ = ('_nBits', '_dist', '_chiral', '_seed', '_cache')

    def __init__(self, index: int, isREWorker: bool):
        nBits: int = _GetSpecialValue_(dFramework['LayeredRDKit_nBits'], index, isREWorker)
        distance: bool = _GetSpecialValue_(dFramework['LayeredRDKit_maxPath'], index, isREWorker)
        super(LayRDK, self).__init__(nBits, distance)

    def GetFpStringTemplate(self) -> str:
        return f'LayRDK {self.GetFpDist()}/#'

    def MolToBitVect(self, mol: Mol):
        return LayeredFingerprint(mol, maxPath=self._dist, fpSize=self._nBits)


class Avalon(BVAbstractEngine):
    # __slots__ = ('_nBits', '_dist', '_chiral', '_seed', '_cache')

    def __init__(self, index: int, isREWorker: bool):
        nBits: int = _GetSpecialValue_(dFramework['Avalon_nBits'], index, isREWorker)
        super(Avalon, self).__init__(nBits, 1)

    def GetFpStringTemplate(self) -> str:
        return f'Avalon #'

    def MolToBitVect(self, mol: Mol):
        return GetAvalonFP(mol, nBits=self._nBits)


class Pattern(BVAbstractEngine):
    # __slots__ = ('_nBits', '_dist', '_chiral', '_seed', '_cache')

    def __init__(self, index: int, isREWorker: bool):
        nBits: int = _GetSpecialValue_(dFramework['Pattern_nBits'], index, isREWorker)
        chiral: bool = _GetSpecialValue_(dFramework['Pattern_Tautomer'], index, isREWorker)
        super(Pattern, self).__init__(nBits, 1, chiral)

    def GetFpStringTemplate(self) -> str:
        return f'Pattern #'

    def MolToBitVect(self, mol: Mol):
        return PatternFingerprint(mol, fpSize=self._nBits, tautomerFingerprints=self._chiral)


class MACCS(BVAbstractEngine):
    # __slots__ = ('_nBits', '_dist', '_chiral', '_seed', '_cache')

    def __init__(self, index: int, isREWorker: bool):
        status: bool = _GetSpecialValue_(dFramework['MACCS'], index, isREWorker)
        super(MACCS, self).__init__(167, 1, status=status)

    def GetFpStringTemplate(self) -> str:
        return f'MACCS #'

    def MolToBitVect(self, mol: Mol):
        return GetMACCSKeysFingerprint(mol)
