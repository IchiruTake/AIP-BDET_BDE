# ----------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license
#  which is included in the file license.txt, found at the root
#  of the Bit2EdgeV2-BDE source tree.
# ----------------------------------------------------------------------------
# This class shares the same behaviour to the :cls:`ModelPlaceholderV1` object,
# but it required our tester to using the same input structure before
# and after cleaning. The model's key and/or model's structure could
# be ignored.
#
# HOWTO: Using the BVManager where you created a list of initial labels
# automatically which is corresponding to the inputConfig.py (We don't want
# to harm the structure here).
# ----------------------------------------------------------------------------

from typing import List, Union, Tuple, Optional, Dict
from Bit2Edge.test.placeholder.SinglePlaceholderV1 import SinglePlaceholderV1


class SinglePlaceholderV2(SinglePlaceholderV1):

    def __init__(self, ModelConfig: Dict[str, str], name: Optional[str] = None):
        super(SinglePlaceholderV2, self).__init__(ModelConfig=ModelConfig, name=name)
        self._weights: Optional[Tuple[float, ...]] = None

    def SetWeights(self, value: Union[List[float], Tuple[float, ...]]) -> None:
        self._weights = tuple(value)

    def GetWeights(self) -> Optional[Tuple[float, ...]]:
        return self._weights

    def GetWeightsSafely(self, num_output: int) -> Tuple[float, ...]:
        weights = self.GetWeights()
        if weights is None or not weights:
            return tuple([0.0] * num_output)
        return weights
