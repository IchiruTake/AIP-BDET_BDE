# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from typing import Any, Dict, Optional, Union

from Bit2Edge.config.ConfigUtils import WriteDictToYamlFile

# [3]: Model Architecture and Configuration
FRAMEWORK: Dict[str, Any] = \
    {
        # ---------------------------------------------
        # // Loss Section //
        'Loss Function': 'Huber', 'Default Loss Keywords in TensorFlow': {'delta': 3.5},
        'Updating Progress': 0.003, 'Unchanged Epochs': 20, 'Frequency': 1,

        # // Fit Section //
        'Shuffle': True,
        'Training Batch Size': 512, 'Testing Batch Size': 512,
        'Initial Epoch': 0, 'Maximum Epochs': 100, 'verbose': True,
        'Package': 'TensorFlow',
        'Workers': 3, 'Queue': 32,

        # ---------------------------------------------
        # // Optimizer Section //
        'Optimizer': 'Adam',
        'Beta_1': 0.9, 'Beta_2': 0.999, 'Momentum': 0,  # Don't adjust these values if you don't know the algorithm
        'Rectify': False, 'Epsilon': 1e-8, 'Weight Decay': 0,  # 10 ** (-4.5),  # Small Value Only: 1e-6 -> 1e-7.

        # If there are optimizer support weight decay, you should check the algorithm:
        # Adam + Weight Decay > 0 --> AdamW
        # But (Adam + Rectify <- True) != RAdam, the optimizer name must be 'RAdam'
        # Don't change these three (3) lines as these are Tensorflow Default.
        'AMSGrad': False, 'centered': False, 'Nesterov': False,
        'MIN_LR': 1e-6, 'Warmup_proportion': 0.1, 'Total Steps': 0, 'SMA': 5.0,
        'Lookahead': False, 'Lookahead-sync-period': 6, 'Lookahead-step-size': 0.5,
    }

MODEL_STRUCTURE: Dict[str, Any] = \
    {
        # --------------------------------------------------------------
        # // BatchNorm Layer //     -> Doesn't have any effect
        'E-model BatchNorm': False, 'BatchNorm-Config': (0.98, 1e-4),

        # // Dropout Layer //       -> Don't enable it. The network is pruned already
        'M-model Dropout': (0, 0, 0),
        'E-model Dropout': (0, 0, 0),  # Unused just for compatibility

        # --------------------------------------------------------------
        # ----------------- // Architecture Section // -----------------
        # G-model
        'G-model Core-Act': 'relu',
        'G-model Last-Act': 'relu',

        # M-model
        'M-model Type': '99r',
        'M-model Core-Act': 'relu', 'M-model Last-Act': None,

        # E-model
        'E-model Type': '99r',
        'E-model Bond-Env Scaling': 2, 'E-model LBondInfo Scaling': 2,
        'E-model Mapping First M-model Layer': True,
        'E-model Core-Branch-Act': 'mish', 'E-model Merge-Act': 'relu', 'E-model Last-Act': None,

        # Don't edit these three keys-values
        'Model Core-Act': 'relu', 'Output Last-Act': None, 'use_bias': True,

        # ----------------- // I/O Layer Section // -----------------
        'Predict-Notion-Index': 0,  # Default must be 0 
        'Predict-Notion': ['BDE', 'BDFE'],
        'Attempt-Sparse': False,
    }

TRAIN_CALLBACKS: Dict[str, Union[str, bool, float, int, Dict[str, Dict]]] = \
    {
        'Callbacks': True,  # Activate any/all callbacks at here
        'Learning Rate Scheduler': True, 'Weight-Decay-Boost': None,
        'Model Checkpoint': True,
        'NaN Encounter': False,
        'Histogram Checking': 'Histogram Profile.csv',
        # Unresolved issue -> Solution: https://github.com/tensorflow/tensorflow/issues/43030#issuecomment-940053771
        'TensorBoard': None,  # datetime.now().strftime('%Y%m%d-%H%M%S'),

        # Please maintain the code here as it is used to standardize the model output name '{epoch:03d}'
        'Checkpoint':
            {
                'TF-Keras Checkpoint {epoch:03d}': {
                    'monitor': 'val_loss',
                    'mode': 'min',
                    'verbose': 0,
                    'save_best_only': True,
                    'save_weights_only': False,
                },
                'TF-Keras Checkpoint [Weight] {epoch:03d}': {
                    'monitor': 'val_loss',
                    'mode': 'min',
                    'verbose': 0,
                    'save_best_only': True,
                    'save_weights_only': True,
                }
            }
    }


def SaveModelConfig(FilePath: str, ModelConfig: Optional[dict] = None) -> None:
    """ Saving the data configuration """
    if ModelConfig is not None:
        return WriteDictToYamlFile(FilePath=FilePath, DataConfiguration=ModelConfig)
    global FRAMEWORK
    return WriteDictToYamlFile(FilePath=FilePath, DataConfiguration=FRAMEWORK)
