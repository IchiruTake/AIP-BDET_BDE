# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from typing import List, Tuple, Union
from rdkit.Chem.rdchem import Mol

from Bit2Edge.config.userConfig import DATA_FRAMEWORK as dFramework
from Bit2Edge.dataObject.InputState import InputState
from Bit2Edge.input.MolProcessor.MolEngine import MolEngine
from Bit2Edge.input.SubgraphUtils import BondPathToMol, AutoSearchBondEnvs
from Bit2Edge.molUtils.molUtils import SmilesFromMol, SmilesToSanitizedMol, CopyAndKekulize


class EnvExtractor:
    """
        This class will separate out the environment out of the molecule,
        using the bond as the center for prediction with defined radius.
        Use method `GetEnvAPI()` to separate the environment.

        Arguments:
        ----------
            - mol (Mol): RDKit molecule
            - bondIdx (int): The bond index/id to define the center

        Returns:
        ----------
            - A list of RDKit molecule
            - An integer defined the bond id at the second-smallest environment
    """

    __slots__ = ('_mode', '_NumEnvs', 'bIdxLocation', '_RE_Pos', '_UseHs',
                 'RecordBondIdxOn2SmallEnv',)

    radius: Tuple[int, ...] = InputState.GetUniqueRadius()

    def __init__(self, RecordBondIdxOn2SmallEnv: bool = False):
        self.RecordBondIdxOn2SmallEnv: bool = RecordBondIdxOn2SmallEnv
        # Function
        # mode=1 -> BS: Bond Searching
        # mode=0 -> RG: Radical Generating; RE: Radical Environment
        self._mode: int = 1

        NumEnvsByRadius: int = InputState.GetNumsInputByRadiusLayer()
        self._NumEnvs: int = InputState.GetNumsInput()
        self._UseHs: Union[bool, Tuple[bool, ...]] = dFramework.get('UseHs', True)

        self.bIdxLocation: int = NumEnvsByRadius - 2  # Second-smallest environment's index: For LBI
        self._RE_Pos: int = NumEnvsByRadius - 1  # First smallest environment's index

    def _GetEnv_(self, mol: Mol, bondIdx: int, paths: List[List[int]]) -> Tuple[List[Mol], int]:
        result: List[Mol] = [0] * self._NumEnvs
        subBondIdx: int = -1
        path_ids = [id(path) for path in paths]
        for idx, path in enumerate(paths):
            if self.RecordBondIdxOn2SmallEnv and idx == self.bIdxLocation:
                result[idx], subBondIdx = BondPathToMol(mol, path, bondIdx=bondIdx)
            else:
                location = path_ids.index(path_ids[idx])    # Find the first index of the path_ids
                if location == idx:
                    result[idx], _ = BondPathToMol(mol, path)
                else:
                    result[idx] = result[location]

        return result, subBondIdx

    def GetEnvsAPI(self, s_mol: Mol, bondIdxes: List[int], safe: bool = False) -> Tuple[List[Mol], int]:
        """
            This is a minor optimization attempts to alleviate the need of double caching available on both
            :var:`s_mol` and :var:`k_mol`. The function first is to generate a kekulized-version
            of the :var:`s_mol`, then iterate all :item:`bondIdx` in :var:`bondIdxes`, and return the sub-mols
            according to the iteration of the :var:`bondIdxes`

            Parameters:
            ----------

            s_mol : Mol
                The sanitized version of the molecule. To have better performance, you should run the function
                :func:`PyCacheEdgeConnectivity` before calling this function

            bondIdxes: List[int]
                The list of bond index applied

            Returns:
            -------

            A list of list of sub-mols and a list of (central) bond index at the second-smallest molecule by radius.

        """
        # [1]: Prepare the molecule

        radius: Tuple[int, ...] = EnvExtractor.radius
        k_mol: Mol = CopyAndKekulize(s_mol)

        # [2]: Prepare the iteration
        # for i, bondIdx in enumerate(bondIdxes):
        #     # [2.1]: Extract all the path
        #     bondPaths: List[List[int]] = SearchBondEnvs(s_mol, radius, bondIdx, useHs=self._UseHs, safe=safe)
        #
        #     # [2.2]: Break to all sub-mols
        #     yield self._GetEnv_(k_mol, bondIdx, bondPaths)

        operation: str = dFramework.get('QueryOperation', 'BFS')
        for i, bondPaths in enumerate(AutoSearchBondEnvs(s_mol, radius, bondIdxes, useHs=self._UseHs,
                                                         operation=operation)):
            yield self._GetEnv_(k_mol, bondIdxes[i], bondPaths)

        pass
