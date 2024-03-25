# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union

from Bit2Edge.config.ConfigUtils import WriteDictToYamlFile, ReadYamlConfigFile

# ----------------------------------------------------------------------------------------------------------------
# [1]: Used for create multi-fingerprint feature
DATA_FRAMEWORK: Dict[str, Union[int, bool, Tuple, str]] = \
    {
        # Basic Descriptors
        'Radius': (6, 4, 2),
        # 'UseHs': (False, True, True),
        'QueryOperation': 'auto',  # 'auto', 'BFS', 'Floyd'
        'LBI_Mode': 1,  # We must ensure that both the LBI_Key and its associated LBI_Labels must be CONSISTENT,

        'Cis-Trans Counting': True, 'Cis-Trans Encoding': True,
        'FP-StereoChemistry': True, 'LBI-StereoChemistry': True,

        # Circular Fingerprint:
        # Extended-Connectivity Fingerprint (Morgan Fingerprint) (ECFP)
        'ECFP_Radius': 2, 'ECFP_nBits': 1024, 'ECFP_Chiral': False,
        # Functional-Class Fingerprint (Variant of Morgan Fingerprint) (FCFP)
        'FCFP_Radius': 3, 'FCFP_nBits': 1024, 'FCFP_Chiral': False,

        # Morgan-Hose ECFP/FCFP Fingerprint (Use outside environment)
        'MHECFP_Radius': 2, 'MHECFP_nBits': 0, 'MHECFP_Chiral': False,
        'MHFCFP_Radius': 3, 'MHFCFP_nBits': 0, 'MHFCFP_Chiral': False,

        'SECFP_Radius': 2, 'SECFP_nBits': 0, 'SECFP_Seed': 0, 'SECFP_Info': (True, True, True),
        # This is min-hashed fingerprint (combination of min-hashed algorithm + ECFP + NLP and Data Mining)
        # Denoted as MHFP6 or SMILES Extended-Connectivity Fingerprint (SECFP)
        # Int Argument contains two main data: number of permutation and hashing seed

        # Substructure Fingerprint
        'RDKit_maxPath': 7, 'RDKit_nBits': 0,  # Daylight Fingerprint (Substructure)
        'LayeredRDKit_maxPath': 7, 'LayeredRDKit_nBits': 0,

        # Path-based Fingerprint
        'AtomPairs_nBits': 1024, 'AtomPairs_Chiral': False,  # Atom-Pairs Fingerprint
        'Torsion_Size': 4, 'Torsion_nBits': 0, 'Torsion_Chiral': False,  # Topological Torsion Fingerprint
        # Activate Tautomer gains more active comparable: This Fingerprints was greedy
        'Pattern_nBits': 0, 'Pattern_Tautomer': False,
        'Avalon_nBits': 0,

        'MACCS': False,  # Although MACCS have 166 bits, implementation of RDKit is 167 where first bit = 0
        # MACCS: https://forum.knime.com/t/rdkit-fingerprint-node-maccs-keys-gives-wrong-number-of-bits/9496

        # Fix name in :class:`LBI_Feat\AtomSymbol`
        'Core-Atoms': (('F', 'Cl', 'Br', 'I', 'P', 'S', 'C', 'H', 'O', 'N'), None),
        # (('UNK', 'F', 'Cl', 'Br', 'I', 'P', 'S', 'C', 'H', 'O', 'N'), 0)
        'Neighbor-Atoms': ((('F', 'Cl'), ('Br', 'I'), ('P', 'S'), 'C', 'H', 'O', 'N'), None),
        # (('UNK', ('F', 'Cl', 'Br', 'I'), ('P', 'S'), 'C', 'H', 'O', 'N'), 0)

        'Cis-Trans Atoms': ('C', 'N'),
        # 26-02-2023 --> Switch to new stereo-chemistry algorithm
        'Legacy': False,  # Stereo-Chemistry: Legacy implementation of RDKit in FindMolChiralCenters

    }


def SaveDataConfig(FilePath: str, DataConfig: Optional[dict] = None) -> None:
    """ Saving the data configuration """
    if DataConfig is not None:
        return WriteDictToYamlFile(FilePath=FilePath, DataConfiguration=DataConfig)
    global DATA_FRAMEWORK
    return WriteDictToYamlFile(FilePath=FilePath, DataConfiguration=DATA_FRAMEWORK)


def UpdateDataConfigFromDict(dictionary: dict, RemoveOldRecord: bool = False) -> None:
    global DATA_FRAMEWORK
    if RemoveOldRecord:
        DATA_FRAMEWORK.clear()
    DATA_FRAMEWORK.update(dictionary)


def UpdateDataConfig(FilePath: str, RemoveOldRecord: bool = False):
    return UpdateDataConfigFromDict(dictionary=ReadYamlConfigFile(FilePath=FilePath), RemoveOldRecord=RemoveOldRecord)
