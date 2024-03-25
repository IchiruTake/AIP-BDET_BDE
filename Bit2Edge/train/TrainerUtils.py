# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This module is to sever as a collection of supporting function for Trainer.py
# --------------------------------------------------------------------------------

from logging import warning, info
from typing import Dict, List, Tuple, Optional

# import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from Bit2Edge.config.devConfig import LR_Wrapper, GetLearningRate
from Bit2Edge.config.modelConfig import FRAMEWORK, TRAIN_CALLBACKS
from Bit2Edge.utils.file_io import FixPath
from Bit2Edge.utils.verify import TestState, InputFullCheck, TestStateByWarning


def _ShowPlot(xlabel: str, ylabel: str, title: str) -> None:
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()
    plt.autoscale()
    plt.show()


def DrawNeuralProgress(training_logs: Dict[str, List[float]], starting: int = 2):
    if starting != 0:
        print(f'The {starting} initial histogram progress has been removed, as the loss are predicted to be '
              f'relatively high which is hard to visualize the detail performance.')
    epochSize: List[int] = list(range(starting, len(training_logs['loss'])))

    plt.clf()
    for key, value in training_logs.items():
        if 'lr' in key:
            continue
        term = 'Training' if 'val' not in key else 'Validation'
        if 'loss' in key:
            plt.plot(epochSize, value[starting:], label=f'{term} loss')
        else:
            metric = key.replace('val_', '').upper()
            plt.plot(epochSize, value[starting:], label=f'{term} metric ({metric})')

    plt.plot(epochSize, [1.0] * len(epochSize), label=f'Acc (1.0)', linestyle='-.')
    plt.plot(epochSize, [2.0] * len(epochSize), label=f'Acc (2.0)', linestyle='-.')
    _ShowPlot(xlabel='Epochs', ylabel='Loss (kcal/mol)', title='Model Progress')

    plt.clf()
    for key, value in training_logs.items(): # Iterate over train_result
        if 'val' in key:
            continue
        val = training_logs.get(f'val_{key}', None)
        if val is None:
            continue
        train = value
        diff = [v - t for v, t in zip(val, train)]
        plt.plot(epochSize, diff[starting:], label='Loss' if 'loss' in key else f'Metric ({key.upper()})')
        if key == 'mse':
            diff = [x ** 0.5 if x > 0 else 0 - (abs(x) ** 0.5) for x in diff]
            plt.plot(epochSize, diff[starting:], label='Loss' if 'loss' in key else f'Metric (RMSE)')

    plt.plot(epochSize, [1.0] * len(epochSize), label=f'Acc (1.0)', linestyle='-.')
    plt.plot(epochSize, [2.0] * len(epochSize), label=f'Acc (2.0)', linestyle='-.')

    _ShowPlot(xlabel='Epochs', ylabel='Loss (kcal/mol)',
              title='Generalization Loss between Training vs Validation')


def GetOptimizer(learning_rate: float = None) -> Optimizer:
    from tensorflow.keras.optimizers import (SGD, Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop)
    name: str = FRAMEWORK['Optimizer'].lower()
    if learning_rate is None:
        learning_rate: float = GetLearningRate(epoch=0)

    epsilon, beta_1, beta_2 = FRAMEWORK['Epsilon'], FRAMEWORK['Beta_1'], FRAMEWORK['Beta_2']
    momentum, weight_decay, rectify = FRAMEWORK['Momentum'], FRAMEWORK['Weight Decay'], FRAMEWORK['Rectify']
    nesterov, amsgrad, centered = FRAMEWORK['Nesterov'], FRAMEWORK['AMSGrad'], FRAMEWORK['centered']
    opt: Optional[Optimizer] = None

    phrase_1: str = 'Optimizer: TF-'
    if name == 'adadelta':
        print(f'{phrase_1}Adadelta')
        opt = Adadelta(learning_rate=learning_rate, epsilon=epsilon)
    elif name == 'adagrad':
        print(f'{phrase_1}Adagrad')
        opt = Adagrad(learning_rate=learning_rate, epsilon=epsilon)
    elif name == 'adamax':
        print(f'{phrase_1}Adamax')
        opt = Adamax(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    elif name == 'nadam' or (name == 'adam' and nesterov is True):
        print(f'{phrase_1}Nesterov-Adam')
        opt = Nadam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    elif name == 'rmsprop':
        print(f'{phrase_1}RMSprop')
        opt = RMSprop(learning_rate=learning_rate, rho=beta_1, centered=centered, momentum=momentum,
                      epsilon=epsilon)

    TF_ADDONS_LOCAL: bool = False
    if opt is None:
        try:
            from Bit2Edge.model.layer import TF_ADDONS
            TF_ADDONS_LOCAL = TF_ADDONS
        except (ImportError, ValueError, ImportWarning, ModuleNotFoundError):
            try:
                import tensorflow_addons
                TF_ADDONS_LOCAL: bool = True
            except (ImportError, ValueError, ImportWarning, ModuleNotFoundError):
                warning('TensorFlow (Add-ons) was not installed. Switch back to Adam.')
                TF_ADDONS_LOCAL: bool = False

    if TF_ADDONS_LOCAL and opt is None:
        phrase_2: str = 'Optimizer: TF-ADDONS-'
        if name in ('sparseadam', 'lazyadam', 'sparse-adam', 'lazy-adam'):
            from tensorflow_addons.optimizers import LazyAdam
            print(f'{phrase_2}LazyAdam')
            opt = LazyAdam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2,
                           amsgrad=amsgrad, epsilon=epsilon)
        elif name == 'sgdw':
            TestState(weight_decay > 0, 'arg::weight_decay must be larger than zero to activate this.')
            from tensorflow_addons.optimizers import SGDW
            print(f'{phrase_2}SGDW')
            opt = SGDW(weight_decay=weight_decay, learning_rate=learning_rate,
                       momentum=momentum, nesterov=nesterov)
        elif name in ('radam', 'r-adam'):
            from tensorflow_addons.optimizers import RectifiedAdam
            print(f'{phrase_2}RAdam')
            opt = RectifiedAdam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                                weight_decay=weight_decay, amsgrad=amsgrad,
                                min_lr=FRAMEWORK['MIN_LR'], warmup_proportion=FRAMEWORK['Warmup_proportion'],
                                total_steps=FRAMEWORK['Total Steps'], sma_threshold=FRAMEWORK['SMA'], )
        elif name == 'adamw':
            TestState(weight_decay > 0, 'arg::weight_decay must be larger than zero to activate this.')
            from tensorflow_addons.optimizers import AdamW
            print(f'{phrase_2}AdamW')
            opt = AdamW(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        elif name == 'adabelief':
            from tensorflow_addons.optimizers import AdaBelief
            print(f'{phrase_2}Adabelief')
            opt = AdaBelief(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2,
                            weight_decay=weight_decay, amsgrad=amsgrad, epsilon=epsilon, rectify=rectify,
                            min_lr=FRAMEWORK['MIN_LR'], warmup_proportion=FRAMEWORK['Warmup_proportion'],
                            total_steps=FRAMEWORK['Total Steps'], sma_threshold=FRAMEWORK['SMA'], )

    if opt is None:
        if name == 'sgd':
            print(f'{phrase_1}SGD')
            opt = SGD(learning_rate=learning_rate, nesterov=nesterov, momentum=momentum)
        elif name == 'adam' or True:  # Enforce Adam as the default optimizer
            print(f'{phrase_1}ADAM')
            opt = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, epsilon=epsilon)

    if FRAMEWORK['Lookahead'] is True:
        if TF_ADDONS_LOCAL:
            from tensorflow_addons.optimizers import Lookahead
            opt = Lookahead(optimizer=opt, sync_period=FRAMEWORK['Lookahead-sync-period'],
                            slow_step_size=FRAMEWORK['Lookahead-step-size'])
            print('>> Optimizer is wrapped by Lookahead in TF_ADDONS.')
        else:
            warning('Optimizer is NOT wrapped by Lookahead in TF_ADDONS.')
    return opt


def GetCallbacks(storage: str, validation_checkpoint: bool = True) -> Optional[List]:
    """
    This function create a list of callbacks for your deep learning model

    Arguments:
    ---------

    storage : str
        The directory folder

    validation_checkpoint : bool
        Whether to use validation set or the train set as the stopping loss (control the Early Stopping).

    Returns:
    -------

    A list of Tensorflow Callbacks
    """
    # Hyper-parameter Verification
    if not TRAIN_CALLBACKS['Callbacks']:
        return None

    if True:
        if storage is None or not isinstance(storage, str):
            warning('In-valid storage. Switch to empty.')
            storage = ''
        else:
            storage = FixPath(FileName=storage, extension='/')

        InputFullCheck(validation_checkpoint, name='validation_checkpoint', dtype='bool')

    CALLBACKS = []
    monitor = 'val_loss' if validation_checkpoint is True else 'loss'
    if 0 < FRAMEWORK['Unchanged Epochs'] <= FRAMEWORK['Maximum Epochs'] - FRAMEWORK['Initial Epoch']:
        from tensorflow.keras.callbacks import EarlyStopping
        CALLBACKS.append(EarlyStopping(monitor=monitor, mode='min', verbose=0, min_delta=FRAMEWORK['Updating Progress'],
                                       patience=FRAMEWORK['Unchanged Epochs']))
    elif FRAMEWORK['Maximum Epochs'] - FRAMEWORK['Initial Epoch'] < FRAMEWORK['Unchanged Epochs']:
        info('Early Stopping does not implement as the number of epochs is not enough.')

    if TRAIN_CALLBACKS['Model Checkpoint'] is True:
        from tensorflow.keras.callbacks import ModelCheckpoint
        TestStateByWarning(len(TRAIN_CALLBACKS['Checkpoint']) <= 2, 'The number of model checkpoints is larger than 2.')

        if len(TRAIN_CALLBACKS['Checkpoint']) <= 2:
            for ckptName, ckpt in TRAIN_CALLBACKS['Checkpoint'].items():
                ckptName = FixPath(ckptName, extension='.h5')
                CKPT = ModelCheckpoint(f'{storage}{ckptName}', monitor=ckpt.get('monitor', 'val_loss'),
                                       verbose=ckpt.get('verbose', 0), save_best_only=ckpt.get('save_best_only', True),
                                       save_weights_only=ckpt.get('save_weights_only', False),
                                       )
                CALLBACKS.append(CKPT)

    if TRAIN_CALLBACKS['Learning Rate Scheduler'] is True:
        from tensorflow.keras.callbacks import LearningRateScheduler
        New_LR_Wrapper = LR_Wrapper
        if TRAIN_CALLBACKS['Weight-Decay-Boost'] is not None:
            wd: float = TRAIN_CALLBACKS['Weight-Decay-Boost'] * FRAMEWORK['Weight Decay']
            New_LR_Wrapper = lambda *args, **kwargs: LR_Wrapper(*args, **kwargs) + wd
        CALLBACKS.append(LearningRateScheduler(schedule=New_LR_Wrapper, verbose=0))

    if TRAIN_CALLBACKS['NaN Encounter'] is True:
        from tensorflow.keras.callbacks import TerminateOnNaN
        CALLBACKS.append(TerminateOnNaN())

    if TRAIN_CALLBACKS['Histogram Checking'] is not None:
        from tensorflow.keras.callbacks import CSVLogger
        CALLBACKS.append(CSVLogger(filename=storage + FixPath(TRAIN_CALLBACKS['Histogram Checking'], extension='.csv')))

    if TRAIN_CALLBACKS['TensorBoard'] is not None:
        from tensorflow.keras.callbacks import TensorBoard
        CALLBACKS.append(TensorBoard(log_dir=storage + TRAIN_CALLBACKS['TensorBoard'], histogram_freq=1,
                                     write_graph=True, write_images=True, write_steps_per_second=False))
    return CALLBACKS


def GetLossAndMetric(loss: str, **kwargs) -> Tuple:
    LOSS, METRIC = None, None
    if loss in ('l1', 'L1', 'mae', 'MAE'):
        LOSS, METRIC = 'mae', 'mse'
    elif loss in ('l2', 'L2', 'mse', 'MSE'):
        LOSS, METRIC = 'mse', 'mae'
    elif loss in ('Huber', 'huber'):
        from tensorflow.keras.losses import Huber
        delta: float = 2.0  # Default set-up for Huber's Loss
        if 'Default Loss Keywords in TensorFlow' in FRAMEWORK:
            delta: float = FRAMEWORK['Default Loss Keywords in TensorFlow'].get('delta', delta)
        LOSS = Huber(delta=kwargs.get('delta', delta))
        METRIC = ['mse', 'mae']
    return LOSS, METRIC


def SaveModel(model: Model, name: Tuple[str, str], SaveOpt: bool = True, save_traces: bool = True) -> None:
    with open(FixPath(FileName=name[0], extension='.json'), 'w') as file:
        file.write(model.to_json())
    model.save_weights(FixPath(FileName=name[1], extension='.h5'), save_format='h5')
    model.save(filepath=FixPath(FileName=name[0], extension='.tf'), save_format='tf',
               include_optimizer=SaveOpt, save_traces=save_traces)
    model.save(filepath=FixPath(FileName=name[0], extension='.h5'), save_format='h5',
               include_optimizer=SaveOpt, save_traces=save_traces)
