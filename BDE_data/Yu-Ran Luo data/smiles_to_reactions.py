from typing import List, Union, Tuple

import pandas as pd

from Bit2Edge.input.MolProcessor.MolDrawer import MolDrawer
from Bit2Edge.input.MolProcessor.MolEngine import MolEngine

def main(smi: str, atoms: Tuple[str, str], isotope: bool = False, neighborAtom: Union[str, List[str]] = None,
         ignoreRadicalWhenDisplay: str = None, AromaticRingAttached: bool = True,
         NonAromaticRingAttached: bool = True, NonRingAttached: bool = True,
         draw: bool = True):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    s = MolEngine()
    mol, reactions = s.GetBond(smi, AtomBondType=atoms, RemoveDuplicate=True, containIsotope=isotope,
                               NeighborAtoms=neighborAtom, StrictNeighborAtomRule=False,
                               AromaticRing=False, NonAromaticRing=False,
                               AromaticRingAttached=AromaticRingAttached,
                               NonAromaticRingAttached=NonAromaticRingAttached,
                               NonRingAttached=NonRingAttached)

    s.Display(reactions, ignoreRadicalWhenDisplay=ignoreRadicalWhenDisplay)
    if draw:
        MolDrawer.DrawReactions(mol=mol, reactions=reactions, filename='mol.png', ImageSize=(1920, 1080), legend=None)
    return s.Export(reactions, filename='mol.csv')

# alkyls = ['C', 'CC', 'CCC', 'C(C)C', 'CCCC', 'C(C)CC', 'CC(C)C', 'C(C)(C)C',
#           'CCCCC', 'C(C)CCC', 'CC(C)CC', 'CCC(C)C', 'C(C)(C)CC', 'C(C)C(C)C',
#           'CC(C)(C)C', 'C(CC)CC', 'CCCCCC']
alkyls = ['C', 'CC', 'CCC', 'C(C)C']
polysulfur = ['S' * i for i in range(1, 13)]
fgroup = {2: '(OC)', 3: '', 4: '(OC)', 5: '', 6: '(OC)'}
k = f''
key1 = f'CCCC'
key2 = f'C(C)(C)C'
nitro = 'N(=O)(=O)'

SMILES = f'CN1CCCN(C)C1COc1ccccc1'
main(smi=SMILES, atoms=('C', 'H'), neighborAtom=[],
     isotope=False, ignoreRadicalWhenDisplay='',
     AromaticRingAttached=True, NonAromaticRingAttached=True,
     NonRingAttached=True, draw=True)
