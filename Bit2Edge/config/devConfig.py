# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from typing import List
from numpy import exp
from datetime import datetime


INIT_LR: float = 0.0015         # Default for initial epoch
LR_LOGS: List[float] = []

def GetLearningRate(epoch: int) -> float:
    if epoch == 0 and len(LR_LOGS) == 0:
        return INIT_LR
    elif epoch == -1:
        return INIT_LR if len(LR_LOGS) == 0 else LR_LOGS[epoch]

    if epoch < 0 or epoch >= len(LR_LOGS):
        raise ValueError('The epoch is out-of-distribution')

    return LR_LOGS[epoch]


def _PostReturnLearningRate(epoch: int, learning_rate: float, verbose: bool) -> float:
    if verbose:
        print(f'\n{datetime.now()} Epoch: {epoch + 1} ---> Learning Rate: {learning_rate:.8f}')
    return learning_rate


def LR_TrainMode(epoch: int, learning_rate: float, verbose: bool) -> float:
    # BoostLR: float = 0.275
    InitDecay: float = -0.0275
    StableDecay: float = -0.01
    EPOCH_DECAYS = [(20, 2.5), (30, 2.5), (50, 4)]
    if epoch != 0:
        for EPOCH_DECAY in EPOCH_DECAYS:
            if epoch == EPOCH_DECAY[0]:
                learning_rate *= (1 / EPOCH_DECAY[1])
                return _PostReturnLearningRate(epoch, learning_rate, verbose)

    if epoch == 0:
        learning_rate = GetLearningRate(epoch=epoch)
    elif epoch <= 6:
        learning_rate *= 1.180
    elif epoch <= 12:
        learning_rate *= exp(InitDecay / 2.5)  # OLD: 2.35
    elif epoch < 35:
        learning_rate *= exp((InitDecay + StableDecay) / 2.50)
    elif epoch < 50:
        learning_rate *= exp(StableDecay / 1.75)
    else:
        learning_rate *= exp(StableDecay / 2.25)

    return _PostReturnLearningRate(epoch, learning_rate, verbose)


def LR_Wrapper(epoch: int, learning_rate: float, verbose: bool = True) -> float:
    if epoch != 0:
        lr = LR_TrainMode(learning_rate=GetLearningRate(epoch=-1) or learning_rate, epoch=epoch, verbose=verbose)
    else:
        lr = GetLearningRate(epoch=epoch)
    lr = float(lr)
    LR_LOGS.append(lr)
    return GetLearningRate(epoch=-1)
