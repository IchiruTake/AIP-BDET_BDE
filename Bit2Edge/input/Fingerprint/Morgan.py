# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem.rdchem import Mol

from Bit2Edge.config.userConfig import DATA_FRAMEWORK as dFramework
from Bit2Edge.input.Fingerprint.FpUtils import BVAbstractEngine, _GetSpecialValue_
from Bit2Edge.utils.verify import TestState


class ECFP(BVAbstractEngine):
    # __slots__ = ('_nBits', '_dist', '_chiral', '_seed', '_cache')

    def __init__(self, index: int, isREWorker: bool):
        nBits: int = _GetSpecialValue_(dFramework['ECFP_nBits'], index, isREWorker)
        radius: int = _GetSpecialValue_(dFramework['ECFP_Radius'], index, isREWorker)
        chiral: bool = _GetSpecialValue_(dFramework['ECFP_Chiral'], index, isREWorker)
        super(ECFP, self).__init__(nBits, radius, chiral)

    def GetFpStringTemplate(self) -> str:
        return f'ECFP {2 * self.GetFpDist()}/#'

    def MolToBitVect(self, mol: Mol):
        return Morgan(mol, useFeatures=False, radius=self._dist, nBits=self._nBits,
                      includeRedundantEnvironments=False, useChirality=self._chiral)


class FCFP(BVAbstractEngine):
    # __slots__ = ('_nBits', '_dist', '_chiral', '_seed', '_cache')

    def __init__(self, index: int, isREWorker: bool):
        nBits: int = _GetSpecialValue_(dFramework['FCFP_nBits'], index, isREWorker)
        radius: int = _GetSpecialValue_(dFramework['FCFP_Radius'], index, isREWorker)
        chiral: bool = _GetSpecialValue_(dFramework['FCFP_Chiral'], index, isREWorker)
        super(FCFP, self).__init__(nBits, radius, chiral)

    def GetFpStringTemplate(self) -> str:
        return f'FCFP {2 * self.GetFpDist()}/#'

    def MolToBitVect(self, mol: Mol):
        return Morgan(mol, useFeatures=True, radius=self._dist, nBits=self._nBits,
                      includeRedundantEnvironments=False, useChirality=self._chiral)


class MHECFP(BVAbstractEngine):
    # __slots__ = ('_nBits', '_dist', '_chiral', '_seed', '_cache')

    def __init__(self, index: int, isREWorker: bool):
        nBits: int = _GetSpecialValue_(dFramework['MHECFP_nBits'], index, isREWorker)
        radius: int = _GetSpecialValue_(dFramework['MHECFP_Radius'], index, isREWorker)
        chiral: bool = _GetSpecialValue_(dFramework['MHECFP_Chiral'], index, isREWorker)
        super(MHECFP, self).__init__(nBits, radius, chiral)

    def GetFpStringTemplate(self) -> str:
        return f'MHECFP {2 * self.GetFpDist()}/#'

    def MolToBitVect(self, mol: Mol):
        return Morgan(mol, useFeatures=False, radius=self._dist, nBits=self._nBits,
                      includeRedundantEnvironments=True, useChirality=self._chiral)


class MHFCFP(BVAbstractEngine):
    # __slots__ = ('_nBits', '_dist', '_chiral', '_seed', '_cache')

    def __init__(self, index: int, isREWorker: bool):
        nBits: int = _GetSpecialValue_(dFramework['MHFCFP_nBits'], index, isREWorker)
        radius: int = _GetSpecialValue_(dFramework['MHFCFP_Radius'], index, isREWorker)
        chiral: bool = _GetSpecialValue_(dFramework['MHFCFP_Chiral'], index, isREWorker)
        super(MHFCFP, self).__init__(nBits, radius, chiral)

    def GetFpStringTemplate(self) -> str:
        return f'MHFCFP {2 * self.GetFpDist()}/#'

    def MolToBitVect(self, mol: Mol):
        return Morgan(mol, useFeatures=True, radius=self._dist, nBits=self._nBits,
                      includeRedundantEnvironments=True, useChirality=self._chiral)


class SECFP(BVAbstractEngine):
    # __slots__ = ('_nBits', '_dist', '_chiral', '_seed', '_cache')

    def __init__(self, index: int, isREWorker: bool):
        nBits: int = _GetSpecialValue_(dFramework['SECFP_nBits'], index, isREWorker)
        radius: int = _GetSpecialValue_(dFramework['SECFP_Radius'], index, isREWorker)
        seed: int = _GetSpecialValue_(dFramework['SECFP_Seed'], index, isREWorker)

        info = dFramework['SECFP_Info']
        TestState(info is not None and len(info) == 3, 'SECFP_Info is unable to be defined.')

        # Special case for SECFP
        if not all(isinstance(information, bool) for information in info):
            info = _GetSpecialValue_(info, index, isREWorker)
            TestState(info is not None and len(info) == 3, 'SECFP_Info is unable to be defined.')
            TestState(all(isinstance(information, bool) for information in info),
                      'All variables found in SECFP_Info must be boolean.')
        if not isinstance(info, tuple):
            info = tuple(info)
        super(SECFP, self).__init__(nBits, radius, seed=seed, info=info, ring=info[0],
                                    isomeric=info[1], kekulize=info[2])

        self._cache['MolToBitVect'] = None
        if self.GetFpStatus():
            from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
            self._cache['MolToBitVect'] = MHFPEncoder(self._cache['ring']).EncodeSECFPMol

    def GetFpStringTemplate(self) -> str:
        return f'SECFP {2 * self.GetFpDist()}/#'

    def MolToBitVect(self, mol: Mol):
        return self._cache['MolToBitVect'](mol, radius=self._dist, length=self._nBits,
                                           rings=self._cache['ring'], isomeric=self._cache['isomeric'],
                                           kekulize=self._cache['kekulize'])
