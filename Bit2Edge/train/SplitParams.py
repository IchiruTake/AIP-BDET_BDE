# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
#  This module is served as an input-controlled parameters for the Dataset Splitting
# --------------------------------------------------------------------------------


class SplitParams:
    __slots__ = ('SplitKey', 'TrainDevSplit', 'mode', 'GoldenRuleForDivision', 'TransferRatio',
                 'sorting', 'objectSorting', 'reverse', 'ZeroDuplicate', 'StrictCleaning',)

    def __init__(self, SplitKey: int = 10) -> None:
        # [1]: Argument to control the overall result
        self.SplitKey: int = SplitKey

        self.TrainDevSplit: bool = False
        self.mode: str = 'sample'
        self.GoldenRuleForDivision: bool = True
        self.TransferRatio: float = 0.5

        # [2.1]: Argument to control the data cleaning (sorting)
        self.sorting: bool = False
        self.objectSorting: bool = False
        self.reverse: bool = False

        # [2.2]: Argument to control the data cleaning (removing duplicate)
        self.ZeroDuplicate: bool = True
        self.StrictCleaning: bool = True
