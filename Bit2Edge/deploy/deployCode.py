# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
from typing import Dict, List, Tuple


# [1]: Model Initialization
SAVED_MODEL_AT_EPOCH: Dict[int, Dict[int, int]] = \
    {
        0: {1: 32, 2: 32, 3: 31},
        1: {1: 31, 2: 32, 3: 33},
        2: {1: 31, 2: 34, 3: 33},
        3: {1: 35, 2: 31, 3: 33},
        42: {1: 32, 2: 32, 3: 33},
    }

# After running Write your weights here to initialize value
SAVED_MODEL_WEIGHTS: Dict[int, Tuple[List[float], Dict[int, List[float]]]] = \
    {
        0: ([0.0588261248565978],
            {
                1: [0.559624315213197],
                2: [0.251527081404306],
                3: [0.188149891277597],
            }
            ),
        1: ([-0.0545227505521169],
            {
                1: [0.449512001190749],
                2: [0.239001992586957],
                3: [0.31204618371449],
            }
            ),
        2: ([-0.0436316354256299],
            {
                1: [0.219670255105031],
                2: [0.517350768571451],
                3: [0.263460622703647],
            }
            ),
        3: ([-0.0361396416039597],
            {
                1: [0.20720356123352],
                2: [0.153259120049858],
                3: [0.64005859425584],
            }
            ),
        42: ([-0.0247125696039685],
             {
                 1: [0.577815892234786],
                 2: [0.344448063591173],
                 3: [0.0779777829946805],
             }
             ),
    }

