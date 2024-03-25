# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from time import perf_counter
from typing import Optional

from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.dataObject.FileParseParams import FileParseParams
from Bit2Edge.dataObject.InputState import InputState
from Bit2Edge.input.Creator import FeatureEngineer
from Bit2Edge.train.DatasetSplitter import DatasetSplitter
from Bit2Edge.train.SplitParams import SplitParams
from Bit2Edge.train.Trainer import Trainer


def ForceActivateGPU(device: int = 0, oneDNN: bool = False, cuBlasFp32: bool = True,
                     GpuMemGrowth: bool = False):
    # TensorFlow GPU enables by default,
    # but we can do extra optimization with OneDNN ?
    # https://medium.com/intel-analytics-software/leverage-intel-deep-learning-optimizations-in-tensorflow-129faa80ee07
    from Bit2Edge.config.startupConfig import OptimizeTensorFlow, EnableDevice
    OptimizeTensorFlow(device=device, oneDNN=oneDNN, cuBlasFp32=cuBlasFp32, GpuMemGrowth=GpuMemGrowth)
    EnableDevice()


def DisplayInfoSet(dataset: FeatureData):
    train_infodata = dataset.GetDataBlock('Info').GetData(environment='Train')
    val_infodata = dataset.GetDataBlock('Info').GetData(environment='Val')
    test_infodata = dataset.GetDataBlock('Info').GetData(environment='Test')
    print('Training Set:', train_infodata.shape, end=' ')
    if val_infodata is not None:
        print('Validation Set:', val_infodata.shape, end=' ')
    if test_infodata is not None:
        print('Testing Set:', test_infodata.shape, end=' ')
    print()


def FeatureSplitter(dataset: FeatureData, ratio: float, seed: int, params: SplitParams) -> None:
    start: float = perf_counter()
    engine = DatasetSplitter(data=dataset)

    engine.Split(ratio=ratio, seed=seed, params=params)

    print(f'Execution Time for Data Processing: {perf_counter() - start:.4f} (s).')
    return DisplayInfoSet(dataset=dataset)


def FeaturePipeline(creator: FeatureEngineer, DoSplitFirst: bool,
                    SplitKey: int, ratio: float, seed: int, mode: str, SplitTrainDevLater: bool,
                    GoldenRuleForDivision: bool, GC: bool, SAFE: bool,
                    QUICK: bool, EvalIfNotSafe: bool, SortOnValTest: bool = True) -> None:
    p = SplitParams(SplitKey=SplitKey)
    p.mode = mode
    p.TrainDevSplit = SplitTrainDevLater
    p.GoldenRuleForDivision = GoldenRuleForDivision

    if DoSplitFirst:
        FeatureSplitter(dataset=creator.Dataset(), ratio=ratio, seed=seed, params=p)

    creator.GetDataAPIs(GC=GC, SAFE=SAFE, SortOnValTest=SortOnValTest)

    if not DoSplitFirst:
        FeatureSplitter(dataset=creator.Dataset(), ratio=ratio, seed=seed, params=p)

    return None

if __name__ == '__main__':
    ENABLE_GPU_IF_DISABLE_BY_TF: bool = True  # TensorFlow GPU enables by default
    if ENABLE_GPU_IF_DISABLE_BY_TF:
        # ForceActivateGPU(device=0, oneDNN=False, cuBlasFp32=True, GpuMemGrowth=False)
        pass

    TIMING: float = perf_counter()
    # ----------------------------------------------------------------------------------------------------------
    # [1.1]: Prepare arguments for Feature Engineering
    SEED: int = 1
    SPLIT_KEY: int = 10
    RUN_TIME: int = 2
    RATIO: float = 0.750
    # DIRECTORY: str = f'resources/BSv1.4-BDE/S{SEED}/R{RUN_TIME}/' # 'resources/TestFolder/'
    DIRECTORY: str = f'resources/testfolder/'  # 'resources/TestFolder/'

    TEST_TO_CONFIGURATION_ONLY = False
    TEST_TO_FEATURE_ENGINEERING_ONLY = True
    TEST_TO_DATA_CLEANING = False
    print(f'Target Directory: {DIRECTORY}')

    # [1.2]: Feature Engineering
    DATASET = FeatureData(trainable=True, retrainable=False)
    TIME_PROFILING: bool = False  # Enable this to debug
    ENGINEER = FeatureEngineer(dataset=DATASET, storeEnvironment=True, TIMER=TIME_PROFILING)

    if TEST_TO_CONFIGURATION_ONLY:
        exit(0)

    LIMIT_SAMPLES: Optional[int] = None
    # 5 if only has one regression target
    FileParams = FileParseParams(mol=0, radical=(1, 2), bIdx=3, bType=4, target=[5])
    ENGINEER.AddCsvData(FilePath='BDE_data/source_dataset_v1.csv', params=FileParams,
                        limitTest=LIMIT_SAMPLES)
    ENGINEER.ExportConfiguration(FilePath='Data Configuration [Before].yaml', StorageFolder=DIRECTORY)

    # [2.1]: Prepare arguments for (Scratch) Model Training
    DATASET.ExportFeatureLabels(Env_FileName=DIRECTORY + '/Env_SavedLabel [Before].csv',
                                LBI_FileName=DIRECTORY + '/LBI_SavedLabel [Before].csv')

    # [2.2]: Split data into Train-Dev-Test <--> Run the Feature-Engineering
    DO_SPLIT_FIRST: bool = True
    GOLDEN_RULE_FOR_DIVISION: bool = True
    SPLIT_MODE: str = 'sample'
    SPLIT_TRAIN_DEV_LATER: bool = False

    UseMol: bool = True
    CleanUp: bool = True
    safe: bool = False
    quick: bool = True
    evalIfNotSafe: bool = False
    SortOnDevTest: bool = True

    FeaturePipeline(creator=ENGINEER, DoSplitFirst=DO_SPLIT_FIRST, SplitKey=SPLIT_KEY, ratio=RATIO, seed=SEED,
                    mode=SPLIT_MODE, SplitTrainDevLater=SPLIT_TRAIN_DEV_LATER,
                    GoldenRuleForDivision=GOLDEN_RULE_FOR_DIVISION, GC=CleanUp, SAFE=safe, QUICK=quick,
                    EvalIfNotSafe=evalIfNotSafe, SortOnValTest=SortOnDevTest)

    DATASET.ExportFeatureLabels(Env_FileName=DIRECTORY + '/Env_SavedLabel [Middle].csv',
                                LBI_FileName=DIRECTORY + '/LBI_SavedLabel [Middle].csv')

    for request, name in (('Train', 'Training'), ('Val', 'Validation'), ('Test', 'Testing')):
        if DATASET.GetDataBlock('Info').GetData(environment=request) is not None:
            continue
        DATASET.ExportInfoToCsv(request=request, FileName=f'{DIRECTORY}/{name} Set [Info].csv', Environment=True)

    if TEST_TO_FEATURE_ENGINEERING_ONLY:
        DATASET.RefreshData()
        exit(0)

    # [2.4]: Post-processing
    DATA_CLEANING: bool = True
    if DATA_CLEANING:
        DATASET.CleanEnvData(EnvFilePath=None, request=DATASET.OptFeatureSetKey())

    ENGINEER.ExportConfiguration(FilePath='Data Configuration [After].yaml', StorageFolder=DIRECTORY)
    DATASET.ExportFeatureLabels(Env_FileName=DIRECTORY + '/Env_SavedLabel [After].csv',
                                LBI_FileName=DIRECTORY + '/LBI_SavedLabel [After].csv')

    # [2.5]: Export Localized Bond Information
    LBI_Name: str = 'LBondInfo_Data'
    if LBI_Name is not None:
        import pandas as pd
        from Bit2Edge.utils.helper import ExportFile

        DF = pd.DataFrame(data=DATASET.GetDataBlock('LBIData').GetData(environment='Train'),
                          columns=DATASET.GetDataBlock('LBIData').GetColumns(), )
        ExportFile(DF, f'{DIRECTORY}{LBI_Name}.csv', index=False)
    # print(f"{label}: {data.shape}")

    SplitTime: float = perf_counter() - TIMING
    print(f'Total Time for Feature Engineering & Data Splitting: {SplitTime:.4f} (s).')

    if TEST_TO_DATA_CLEANING:
        DATASET.RefreshData()
        exit(0)

    # ----------------------------------------------------------------------------------------------------------
    # """ <<- Break here for Feature Engineering Only
    Trainer = Trainer(dataset=DATASET, StorageFolder=DIRECTORY)
    Trainer.SaveModelConfig(f'Model Configuration [Before].yaml', StorageFolder=None)

    # [2.6]: (Scratch) Model Training
    MODEL_KEY: str = '03'
    PRETRAINED_MODEL: Optional[str] = None
    Trainer.InitializeModel(ModelKey=MODEL_KEY, TF_Model=PRETRAINED_MODEL)
    Trainer.SaveModelConfig(f'Model Configuration [Middle].yaml', StorageFolder=None)

    BEST_VALIDATION_MODEL: bool = True
    # For RG-BDE, duplicated samples is removed after train
    Trainer.TrainAPI(useBestValidation=BEST_VALIDATION_MODEL, SaveOpt=True)
    Trainer.SaveModelConfig(f'Model Configuration [After].yaml', StorageFolder=None)
    # ----------------------------------------------------------------------------------------------------------
    # """
    print('-' * 81)
    print(f'Total Time for Feature Engineering & Data Splitting: {SplitTime:.4f} (s).')
    print(f'Measured Time of train.py: {perf_counter() - TIMING:.4f} (s).')
