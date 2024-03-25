# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
from typing import Optional


class PredictParams:
    __slots__ = ('getLastLayer', 'average', 'ensemble', 'force', 'mode', 'Sfs', 'verbose')

    def __init__(self):
        self.getLastLayer: bool = False
        self.average: bool = True
        self.ensemble: bool = True
        self.force: bool = False
        self.mode: int = 2
        self.Sfs: Optional[int] = None
        self.verbose: bool = False
