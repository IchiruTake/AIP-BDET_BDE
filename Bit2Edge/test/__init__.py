# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from logging import warning

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except (ImportError, ImportWarning, RuntimeError):
    warning('Scikit-Learn Intel OpenMP Extension is not installed.')
