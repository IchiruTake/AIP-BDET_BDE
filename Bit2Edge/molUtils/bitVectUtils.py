# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from numpy import frombuffer, ndarray, uint8


def StringVectToNumpy(bitVect: str) -> ndarray:
    """ This function converted the bitVect in bit-string datatype to numpy array. """
    return frombuffer(bitVect.encode(), dtype=uint8) - 48
