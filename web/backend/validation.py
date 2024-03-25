# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
from collections import defaultdict
from logging import warning
from typing import Optional, Dict

from rdkit.Chem.rdchem import Mol

import web.config.config as CONF
from Bit2Edge.molUtils.molUtils import SmilesToSanitizedMol, CanonMolString, PyCacheEdgeConnectivity, PyGetAtoms

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
DEFAULT_MOL_MODE = CONF.GetDeploymentVariable('DEFAULT_MOL_MODE')


# [1]: SMILES Validator
def GetCanonicalSmiles(smiles: str, mode: str = DEFAULT_MOL_MODE, ignore_error: bool = False) -> Optional[str]:
    try:
        return CanonMolString(smiles, mode=mode, useChiral=True)
    except (RuntimeError, ValueError, TypeError) as err:
        if not ignore_error:
            raise err
    return None


def IsValidSmiles(smiles: str, mode: str = DEFAULT_MOL_MODE) -> bool:
    return GetCanonicalSmiles(smiles=smiles, mode=mode) is not None


def LoadMolInTrainScope(smiles: str, mode: str = DEFAULT_MOL_MODE) -> Optional[dict]:
    if not IsValidSmiles(smiles=smiles, mode=mode):
        return None
    TRAIN_RESULT = CONF.GetDeploymentVariable('TRAIN_FILE', 'RESULT_CONFIG')
    result = TRAIN_RESULT.get(smiles, None)
    if result is not None:
        return result

    warning('The :arg:`smiles` has been attempted to canonicalize for searching')
    canonical_smiles = GetCanonicalSmiles(smiles=smiles, mode=mode)
    result = TRAIN_RESULT.get(canonical_smiles, None)
    return result


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# [2]: Molecule evaluation (From this part, ensure the molecule must be valid)
def EvalAtomInMol(smiles: str, train: bool = True, non_train: bool = True) -> dict:
    if not train and not non_train:
        return {}
    smiles = GetCanonicalSmiles(smiles=smiles, mode=DEFAULT_MOL_MODE, ignore_error=False)
    mol: Mol = SmilesToSanitizedMol(smiles)
    PyCacheEdgeConnectivity(mol)
    result = {
        'train': defaultdict(lambda: 0),
        'non_train': defaultdict(lambda: 0)
    }
    SCOPE = CONF.GetDeploymentVariable('TRAIN_FILE', 'SCOPE')
    for atom in PyGetAtoms(mol):
        atom_symbol = atom.GetSymbol()
        IN_SCOPE = (atom_symbol in SCOPE)
        if IN_SCOPE and train:
            result['train'][atom_symbol] += 1
        elif not IN_SCOPE and non_train:
            result['non_train'][atom_symbol] += 1
    return result
