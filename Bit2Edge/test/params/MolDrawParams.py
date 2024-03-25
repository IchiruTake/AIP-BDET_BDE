# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module serves as a input-attached class to manage all arguments for
# drawing the molecule.
# --------------------------------------------------------------------------------


from typing import Optional, Tuple

from Bit2Edge.utils.verify import InputCheckRange, InputFullCheck, InputCheckIterable


class MolDrawParams:
    __slots__ = ('ImageSize', 'Sfs', 'NumWeakestTarget', 'SortOnTarget', 'NameImageByStartRow',
                 'BreakDownBondImage', 'DenyMolImageFile')

    def __init__(self):
        self.ImageSize: Tuple[int, int] = (1080, 720)
        self.Sfs: int = 2
        self.SortOnTarget: bool = False
        self.NumWeakestTarget: Optional[int] = None
        self.NameImageByStartRow: bool = False
        self.BreakDownBondImage: bool = False
        self.DenyMolImageFile: bool = False
        self.evaluate()

    def evaluate(self):
        InputCheckRange(self.NumWeakestTarget, name='NumWeakestTarget', maxValue=20, minValue=1,
                        allowNoneInput=True, leftBound=True, rightBound=True)
        InputFullCheck(self.NameImageByStartRow, name='NameImageByStartRow', dtype='bool')
        InputFullCheck(self.BreakDownBondImage, name='BreakDownBondImage', dtype='bool')
        InputFullCheck(self.DenyMolImageFile, name='DenyMolImageFile', dtype='bool')
        InputFullCheck(self.Sfs, name='Sfs', dtype='int')
        InputCheckIterable(value=self.ImageSize, name='ImageSize', maxValue=2 ** 16 - 1, minValue=0)
