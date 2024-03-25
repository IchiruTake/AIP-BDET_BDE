# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module stored some utility functions for molecule processing
# --------------------------------------------------------------------------------

from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.rdMolDraw2D import PrepareAndDrawMolecule
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem.rdmolops import AddHs

from Bit2Edge.input.MolProcessor.MolEngine import MolEngine
from Bit2Edge.input.MolProcessor.utils import _LABEL
from Bit2Edge.molUtils.molUtils import SmilesToSanitizedMol, SmilesFromMol
from Bit2Edge.utils.verify import TestState


class MolDrawer:
    @staticmethod
    def _BaseDrawToolSetup_(params: rdMolDraw2D.MolDrawOptions):
        params.centreMoleculesBeforeDrawing = True
        params.dummyIsotopeLabels = False
        params.includeRadicals = False
        params.clearBackground = True
        params.dummiesAreAttachments = False
        params.padding = 0.05

    @staticmethod
    def _SetDrawToolCairo_(tool: Draw.MolDraw2DCairo, ImageSize: Tuple[int, int]):
        params = tool.drawOptions()
        MolDrawer._BaseDrawToolSetup_(params)

        FontSize = tool.FontSize()
        if ImageSize[0] * ImageSize[1] >= int(4e6):
            tool.SetFontSize(0.70 * FontSize)
            params.legendFontSize = 400
            params.annotationFontScale = 0.8
        elif ImageSize[0] * ImageSize[1] >= int(2e6):
            tool.SetFontSize(0.65 * FontSize)
            params.legendFontSize = 250
            params.annotationFontScale = 0.75
        else:
            tool.SetFontSize(0.6 * FontSize)
            params.legendFontSize = 100
            params.annotationFontScale = 0.70

        return

    @staticmethod
    def DrawMol2DCairo(mol: RWMol, mapping: Dict[int, str], filename: str = 'mol.png',
                       ImageSize: Tuple[int, int] = (1080, 720), legend: Optional[str] = '') -> None:
        # Warning: This is compatible with version 2021.09.xx or lower
        # https://greglandrum.github.io/rdkit-blog/technical/2022/03/18/refactoring-moldraw2d.html
        tool: Draw.MolDraw2DCairo = Draw.MolDraw2DCairo(*ImageSize)
        MolDrawer._SetDrawToolCairo_(tool, ImageSize=ImageSize)

        if mol.GetNumConformers() == 0:
            rdDepictor.Compute2DCoords(mol)

        if legend is None:
            legend = f'Mol: {SmilesFromMol(mol, keepHs=False)}'

        try:
            tool.DrawMolecule(mol, highlightBonds=[*mapping], legend=legend)
        except (RuntimeError, ValueError):
            PrepareAndDrawMolecule(tool, mol, legend=f'Mol: {legend}')
        tool.FinishDrawing()
        if filename:
            tool.WriteDrawingText(filename)
        plt.show()
        tool.ClearDrawing()
        return None

    # ================================================================================================================
    @staticmethod
    def _SetDrawToolSVG_(tool: rdMolDraw2D.MolDraw2DSVG):
        tool.SetFontSize(0.65)
        params = tool.drawOptions()
        MolDrawer._BaseDrawToolSetup_(params)

        # Additional parameters
        params.fixedBondLength = 30
        params.highlightBondWidthMultiplier = 20

    @staticmethod
    def DrawMol2DSVG(smiles: str, mapping: Dict[int, str], ImageSize: Tuple[int, int] = (300, 300),
                     legend: Optional[str] = '') -> str:
        # Warning: This is compatible with version 2021.09.xx or lower
        # https://greglandrum.github.io/rdkit-blog/technical/2022/03/18/refactoring-moldraw2d.html
        # https://github.com/NREL/alfabet/blob/master/alfabet/drawing.py

        # [1]: Run preparation
        mol_no_H = SmilesToSanitizedMol(smiles, addHs=False)
        TestState(mol_no_H is not None, f'Incorrect SMILES: {smiles}')
        TestState(len(mapping) == 1, 'Only accept one mapping value.')
        bond_index = [*mapping][0]
        print(f'Current bond index: {bond_index} --> Mapping: {mapping}')
        if bond_index >= mol_no_H.GetNumBonds():
            molH = AddHs(mol_no_H)
            if bond_index >= molH.GetNumBonds():
                raise RuntimeError(f'Fewer than {bond_index} bonds in {smiles}: {molH.GetNumBonds()} total bonds.')
            bond = molH.GetBondWithIdx(bond_index)

            start_atom = mol_no_H.GetAtomWithIdx(bond.GetBeginAtomIdx())
            mol = AddHs(mol_no_H, onlyOnAtoms=[start_atom.GetIdx()])
            new_bond_index = mol.GetNumBonds() - 1

            # Update mapping
            mapping[new_bond_index] = mapping.pop(bond_index)
            bond_index = new_bond_index
            print(f'\t>> Updated bond index: {bond_index} --> Mapping: {mapping}')
        else:
            mol = mol_no_H

        # mol = RWMol(mol)
        if mol.GetNumConformers() == 0:
            rdDepictor.Compute2DCoords(mol)

        # mol.GetBondWithIdx(bond_index).SetProp('bondNote', mapping[bond_index])

        # [2]: Draw molecule
        tool = rdMolDraw2D.MolDraw2DSVG(*ImageSize)
        MolDrawer._SetDrawToolSVG_(tool)
        if legend is None:
            legend = f'Mol: {SmilesFromMol(mol, keepHs=False)}'
        highlightAtoms = [mol.GetBondWithIdx(bond_index).GetBeginAtomIdx(),
                          mol.GetBondWithIdx(bond_index).GetEndAtomIdx()]

        tool.DrawMolecule(mol, highlightAtoms=highlightAtoms, highlightBonds=[*mapping], legend=legend)
        tool.FinishDrawing()
        svg_image = tool.GetDrawingText()
        plt.show()
        tool.ClearDrawing()
        return svg_image

    @staticmethod
    def DrawReactions(mol: RWMol, reactions: List, filename: Optional[str] = 'mol.png',
                      ImageSize: Tuple[int, int] = (1080, 720), legend: Optional[str] = None) -> None:
        idx: int = _LABEL.index('Canonical')
        mols = [reaction[idx] for reaction in reactions]
        TestState(len(set(mols)) == 1, msg='There are more than one molecule or no molecule detected.')
        mapping = {reaction[-2]: f'{reaction[-2]}: {reaction[-1]}' for reaction in reactions}

        new_mol = MolEngine.EmbedBondNote(mol=mol, mapping=mapping, inplace=False)
        return MolDrawer.DrawMol2DCairo(new_mol, mapping=mapping, filename=filename, ImageSize=ImageSize,
                                        legend=legend)
