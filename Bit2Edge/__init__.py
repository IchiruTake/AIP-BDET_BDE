#
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license
#  which is included in the file license.txt, found at the root
#  of the Bit2EdgeV2-BDE source tree.
#

import gc

__VERSION__: str = '0.0.1'
__TEST_VERSION__: str = '-alpha.01'


def __version__():
    return __VERSION__


def __test_version__():
    return __TEST_VERSION__


def __full_version__():
    return f'{__version__()}{__test_version__()}'


if not gc.isenabled():
    gc.enable()
