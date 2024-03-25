# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from Bit2Edge.utils.verify import TestState


def TestSafeTester(func):
    def TestSafeDataset(*args, **kwargs):
        tester = args[0]
        msg: str = 'is/are in-valid and vulnerable.'
        t_dataset = tester.Dataset()
        TestState(t_dataset.OptFeatureSetKey() == 'Test', f'This object::dataset {msg}')
        TestState(t_dataset is tester.Generator().Dataset(),
                  f'The two datasets inside tester and generator {msg}')
        return func(*args, **kwargs)

    return TestSafeDataset
