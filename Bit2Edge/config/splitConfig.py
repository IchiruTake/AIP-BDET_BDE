# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from typing import Dict, Tuple

# [1]: Framework for splitting data
SPLIT_FRAMEWORK: Dict[int, Tuple[Tuple[bool, bool, bool], bool, str]] = \
    {
        -1: ((True, False, False), False, 'Training Set Only'),

        # Two separate factor
        0: ((True, False, True), False, 'Training & Testing Set'),
        1: ((True, True, False), False, 'Training & Held-out Validation Set'),
        2: ((True, True, False), True, 'Training & Sample-Trained Validation'),

        # Three separate factor
        10: ((True, True, True), False, 'Training & Held-out Validation & Testing Set. 2 train_test_split(s)'),
        11: ((True, True, True), False, 'Training & Held-out-bottom Validation & Testing Set. 1 train_test_split'),
        12: ((True, True, True), False, 'Training & Held-out-top Validation & Testing Set. 1 train_test_split'),

        15: ((True, True, True), True, 'Training & Sample-Trained Validation & Testing Set. 2 train_test_split(s)'),
        16: ((True, True, True), True, 'Training & Sample-Trained-bottom Validation & Testing Set. 1 train_test_split'),
        17: ((True, True, True), True, 'Training & Sample-Trained-top Validation & Testing Set. 1 train_test_split'),
    }


def ValidateTrainingKey(SplitKey: int) -> None:
    from Bit2Edge.utils.helper import InputFullCheck
    if SplitKey not in SPLIT_FRAMEWORK:
        raise TypeError('This key cannot be found in the configuration.')

    value = SPLIT_FRAMEWORK[SplitKey]
    if len(value) != 3:
        raise ValueError(f'This key (={SplitKey}) should have 3 instead of {len(value)} values.')
    if len(value[0]) != 3:
        raise ValueError(f'This key[0] (={SplitKey}) should have 3 instead of {len(value)} values.')
    InputFullCheck(value[0][0], name='TrainingStatus', dtype='bool')
    InputFullCheck(value[0][1], name='ValidationStatus', dtype='bool')
    InputFullCheck(value[0][2], name='TestingStatus', dtype='bool')

    InputFullCheck(value[1], name='Held-out Validation', dtype='bool')
    InputFullCheck(value[2], name='Description', dtype='str')

    if value[1] and not value[0][1]:
        raise TypeError('If the validation set is not available, the held-out validation status should be False.')
    return None


def ValidateEveryTrainingKey() -> None:
    for key, _ in SPLIT_FRAMEWORK.items():
        ValidateTrainingKey(SplitKey=key)
    return None


def GetTrainingState(TrainingKey: int) -> Tuple[bool, bool, bool]:
    return SPLIT_FRAMEWORK[TrainingKey][0]


def IsHeldOutValidation(TrainingKey: int) -> bool:
    if SPLIT_FRAMEWORK[TrainingKey][1] and not SPLIT_FRAMEWORK[TrainingKey][0][1]:
        raise TypeError('If the validation set is not available, the held-out validation status should be False.')
    # It can be `return not SPLIT_FRAMEWORK[SplitKey][1]` but to guarantee consistency
    return not SPLIT_FRAMEWORK[TrainingKey][1]


def GetTrainingKeyDescription(TrainingKey: int) -> str:
    result = SPLIT_FRAMEWORK[TrainingKey][2]
    print(result)
    return result
