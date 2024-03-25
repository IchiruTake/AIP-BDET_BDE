# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from typing import Dict, List, Optional, Tuple, Union

from rdkit.Chem.rdchem import Mol

from Bit2Edge.config.userConfig import DATA_FRAMEWORK as dFramework
from Bit2Edge.input.LBI_Feat.AtomBondV1 import AtomHybridV1, AtomRingV1, BondStateV1
from Bit2Edge.input.LBI_Feat.AtomBondV2 import AtomHybridV2, AtomRingV2, AtomChargeV2, \
    BondTypeV2_01, BondTypeV2_02, BondRingV2
from Bit2Edge.input.LBI_Feat.LBI_Feat import AtomSymbol
from Bit2Edge.input.LBondInfoUtils import (CountCisTrans, DetectCisTrans, FindStereoAtomsBonds, GetCisTransLabels,
                                           GetMolStereo, GetStereoChemistry, _AddMulti_,
                                           _AddSingle_, _WarnStereochemistry_)
from Bit2Edge.molUtils.molUtils import PyGetBondNeighbors, PyCacheEdgeConnectivity, PyGetBondWithIdx, PyGetBonds, \
    IsNotObsoleteAtom
from Bit2Edge.utils.verify import TestState

STR: Union[str, Tuple[str, ...], List[str]]


class LBICreator:
    """
        This class is the last component which convert a specified bond in
        a given molecule into a feature of vector containing its localized
        information (localized bond information, or LBondInfo, or LBI).
        This is achieved by method `self.GetLBondInfoAPI()`

        Arguments:
        ----------
            - mol (Mol): A RDKit molecule
            - bondIdx (int): The bond id we want to predict

        Returns:
        -------
            - The feature of vector is represented as a list of integer
    """

    __slots__ = ('_CoreLabels', '_BondTypeLabels', '_CisTransLabels',
                 'AtomTokenDict', 'AtomTokenMaxLength', 'AtomFuncs',
                 'BondTokenDict', 'BondTokenMaxLength', 'BondFuncs',
                 '_CacheState', '_InitLBIStackSize',
                 '_CisTranCount', '_CisTranEnc', '_StereoChemistry',
                 )

    def __init__(self, mode: Optional[int] = None, StereoChemistry: bool = True):
        """
            This is the feature generator to create localized bond information

            [Old]: Atom: (AtomSymbols, Hybrids, AtomRings) --- Bond: NeighborBondList
            [New]: Atom: (AtomSymbols, Hybrids, AtomRings) --- Bond: BondType, BondStatus

            Step 2: Ordering: Associated Atoms --> Neighbor Atoms --> Target Bonds
            (i): Label Features at self.CountingLabels
            (ii): Filter out same operation at self.tokenDict (atom-neighbor: counting ~ Tuple)
            (iii): Locate the location of self.tokenLength. First value is constraint between
            atom-neighbor: counting & bond counting and identifier.
        """
        if mode is None:
            mode: int = dFramework['LBI_Mode']
        TestState(mode in (0, 1), 'This mode for setting is not supported currently or under development.')

        # -----------------------------------------------------------------------------------------
        length: int = 0
        self._CoreLabels: List[str] = []

        self.AtomTokenDict: List[Dict[str, Union[int, Tuple[int, int]]]] = []
        self.AtomTokenMaxLength: List[Union[int, Tuple[int, int]]] = []

        self.BondTokenDict: List[Dict[str, Union[int, Tuple[int, int]]]] = []
        self.BondTokenMaxLength: List[Union[int, Tuple[int, int]]] = []

        # Result in here:
        if mode == 0:
            ENGINES = (AtomSymbol(isNeighbor=False), AtomHybridV1(), AtomRingV1())
            AtomFeatures = tuple(engine.GetLabel() for engine in ENGINES)
            self.AtomFuncs: Tuple = (tuple(engine.AtomBondToMol for engine in ENGINES),
                                     tuple(engine.AtomBondToMol for engine in ENGINES))
            length = _AddMulti_(AtomFeatures, AtomFeatures, size=length, labels=self._CoreLabels,
                                tables=self.AtomTokenDict, lengths=self.AtomTokenMaxLength)
        elif mode == 1:
            # rdHybrid.UNSPECIFIED == rdHybrid.OTHER ????
            ENGINES_01 = (AtomSymbol(isNeighbor=False), AtomHybridV2(), AtomRingV2(), AtomChargeV2())
            FtList1 = tuple(engine.GetLabel() for engine in ENGINES_01)

            ENGINES_02 = (AtomSymbol(isNeighbor=True), AtomHybridV2(), AtomRingV2(), AtomChargeV2())
            FtList2 = tuple(engine.GetLabel() for engine in ENGINES_02)

            self.AtomFuncs: Tuple = (tuple(engine.AtomBondToMol for engine in ENGINES_01),
                                     tuple(engine.AtomBondToMol for engine in ENGINES_02))
            length = _AddMulti_(FtList1, FtList2, size=length, labels=self._CoreLabels,
                                tables=self.AtomTokenDict, lengths=self.AtomTokenMaxLength)
        else:
            raise ValueError('The configuration mode is undefined.')

        # [1.2.2]: Bond Features
        if mode == 0:
            ENGINE = BondStateV1()
            self.BondFuncs: Tuple = (tuple(), (ENGINE.AtomBondToMol,))
            length = _AddSingle_(ENGINE.GetLabel(), size=length, labels=self._CoreLabels,
                                 tables=self.BondTokenDict, length=self.BondTokenMaxLength)
        elif mode == 1:
            # See here: https://www.quora.com/Why-dative-bond-is-shown-by-double-bond
            ENGINES_01 = (BondTypeV2_01(), BondRingV2())
            ENGINES_02 = (BondTypeV2_02(), BondRingV2())
            self.BondFuncs: Tuple = (tuple(engine.AtomBondToMol for engine in ENGINES_01),
                                     tuple(engine.AtomBondToMol for engine in ENGINES_02))
            FtList1 = tuple(engine.GetLabel() for engine in ENGINES_01)
            FtList2 = tuple(engine.GetLabel() for engine in ENGINES_02)
            length = _AddMulti_(FtList1, FtList2, size=length, labels=self._CoreLabels,
                                tables=self.BondTokenDict, lengths=self.BondTokenMaxLength)

        # [1.3]: Lock the result
        self.AtomTokenDict: Tuple = tuple(self.AtomTokenDict)
        self.AtomTokenMaxLength: Tuple = tuple(self.AtomTokenMaxLength)
        self.BondTokenDict: Tuple = tuple(self.BondTokenDict)
        self.BondTokenMaxLength: Tuple = tuple(self.BondTokenMaxLength)

        # ---------------------------------------------------------------------------------------------------
        # Cis-Trans
        self._StereoChemistry: bool = StereoChemistry

        self._CisTransLabels: List[str] = GetCisTransLabels()
        self._CisTranEnc: bool = dFramework['Cis-Trans Encoding']
        self._CisTranCount: bool = dFramework['Cis-Trans Counting']

        # Cache:
        self._InitLBIStackSize: int = len(self._CoreLabels) + 1
        self._CacheState: Dict[str, bool] = {}

    # ----------------------------------------------------------------------------------------------------------
    # Graph traversing &
    def UpdateNewMol(self, mol: Mol, NoneForStereo: bool = False) -> None:
        mol.StereoState = 1
        mol.EdgeStereo = (None, None)
        PyCacheEdgeConnectivity(mol)
        PyGetBonds(mol)  # Cache all bonds (SHOULD be behind the PyCacheConnectivity)
        if self._StereoChemistry or self._CisTranCount or self._CisTranEnc:
            mol.EdgeStereo = GetMolStereo(mol, useNone=NoneForStereo)  # Reset cache
            if self._StereoChemistry and not GetStereoChemistry(mol, MolStereoR=mol.EdgeStereo[1]):
                mol.StereoState = 0
        return None

    def GetLBondInfoAPI(self, mol: Mol, bondIdx: int, UpdateMol: bool, safe: bool = False) -> List[int]:
        """
        This method run a consistent pipeline to convert from an argument of molecule(s) to feature. 
        The pipeline is called by the following order:

        1) Initialize the result
        2) Generate the localized bond information (atom-neighbor-bond)
        3) Determine the stereo-chemistry (Push cache ahead)
        4) Determine the cis-trans

        Arguments:
        ---------

        mol : Mol
            The RDKit molecule to be considered
        
        bondIdx : int
            The bond index to be considered in the molecule
        
        UpdateMol : bool, optional
            If either True or None, the stereo-chemistry is calculated all the time. This is a 
            user-defined cache to reduce time spent on the stereo-chemistry assignment. Note 
            that if the molecule is cached by :arg:`mol.StereoState` or :arg:`mol.EdgeStereo`,
            unless :arg:`UpdateMol` is True, the stereo-chemistry will not be calculated. 

        safe : bool
            If True, the function will make bond traversal on every atom iteration. Default 
            to False.

        Returns:
        -------
        
        A list of integer contained the localized bond information.
        """
        # [1]: Initialize the result
        NONE: bool = False
        if UpdateMol is True or not hasattr(mol, 'EdgeStereo'):  # Speed-up by this order
            FindStereoAtomsBonds(mol, NeedAssignStereochemistry=self._StereoChemistry)
            self.UpdateNewMol(mol, NoneForStereo=NONE)

        MolStereo, MolStereoR = mol.EdgeStereo

        # [2]: Generate the localized bond information (atom-neighbor-bond)
        stack = self.LBI_Function(mol, bondIdx, safe=safe)

        # [3]: Determine the stereo-chemistry (Push cache ahead)
        # Use cache to reduce stereo-chemistry complexity. Code equivalent
        # if self._StereoChemistry and GetStereoChemistry(mol, MolStereoR=MolStereoR):
        #   stack[self._InitLBIStackSize - 1] = 1
        if self._StereoChemistry and mol.StereoState == 1:
            stack[self._InitLBIStackSize - 1] = 1

        # [4]: Determine the cis-trans
        if MolStereo:
            # This method is used to determine the cis-trans configuration in the system.
            if self._CisTranCount:  # Count Z-Bonds & Count E-bonds
                # CisTransBond.count("E") + CisTransBond.count("Z") = len(CisTransBond)
                # https://www.rdkit.org/docs/cppapi/Bond_8h_source.html Line 274
                stack.extend(CountCisTrans(MolStereoR))

            if self._CisTranEnc:
                DetectCisTrans(mol, bondIdx, stack, MolStereo, useNone=NONE, safe=safe)

        return stack

    # ----------------------------------------------------------------------------------------------------------
    def LBI_Function(self, mol: Mol, bondIdx: int, safe: bool) -> List[int]:
        # [1]: Initialize the vector
        stack: List[int] = [0] * self._InitLBIStackSize
        bond = PyGetBondWithIdx(mol, bondIdx)

        CORE_ATOM_OPS, NBR_ATOM_OPS = self.AtomFuncs
        CORE_BOND_OPS, NBR_BOND_OPS = self.BondFuncs

        # [2.1.1]: Evaluate the state of the current bond (Order independent).
        for idx, func in enumerate(CORE_BOND_OPS):
            func(bond, stack, 0, idx, positional_datamap=self.BondTokenDict,
                 length_datamap=self.BondTokenMaxLength)

        # [2.1.2]: Evaluate the state of the two current atoms (Order independent).
        for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            for idx, func in enumerate(CORE_ATOM_OPS):
                func(atom, stack, 0, idx, positional_datamap=self.AtomTokenDict,
                     length_datamap=self.AtomTokenMaxLength)

            if safe or IsNotObsoleteAtom(mol, atom=atom):
                for neighbor_bond in PyGetBondNeighbors(mol, atom):
                    if neighbor_bond.GetIdx() == bondIdx:
                        continue
                    neighbor_atom = neighbor_bond.GetOtherAtom(atom)

                    # [2.2.1]: Evaluate the state of the neighboring atoms (Order independent).
                    for idx, func in enumerate(NBR_ATOM_OPS):
                        func(neighbor_atom, stack, 1, idx, positional_datamap=self.AtomTokenDict,
                             length_datamap=self.AtomTokenMaxLength)

                    # [2.2.2]: Evaluate the state of the neighboring bond (Order independent).
                    for idx, func in enumerate(NBR_BOND_OPS):
                        func(neighbor_bond, stack, 1, idx, positional_datamap=self.BondTokenDict,
                             length_datamap=self.BondTokenMaxLength)
        return stack

    # ---------------------------------------------------------------------------------------------------------------
    def GetLabels(self) -> List[str]:
        return self.GetCoreLabels() + self.GetCisTransLabels()

    def GetCoreLabels(self) -> List[str]:
        return self._CoreLabels

    def GetCisTransLabels(self) -> List[str]:
        return self._CisTransLabels

    def CheckStereochemistry(self, mol: Mol, LBI_Vect: List[int],
                             previous_message: Optional[str] = None) -> Optional[str]:
        if LBI_Vect[len(self._CoreLabels)] == 0:  # First value of self._CisTransLabels
            return _WarnStereochemistry_(mol, previous_message=previous_message)
        return previous_message
