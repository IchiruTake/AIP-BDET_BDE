# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
from time import perf_counter

import pandas as pd

from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.dataObject.FileParseParams import FileParseParams
from Bit2Edge.deploy.deploy import DEPLOYMENT
from Bit2Edge.input.Creator import FeatureEngineer
from Bit2Edge.test.GroupTester import GroupTester
# from Bit2Edge.test.TesterUtilsP1 import DisplayDataFrame
from Bit2Edge.test.params.PredictParams import PredictParams
from Bit2Edge.utils.file_io import ExportFile

# --------------------------------------------------------------------------------
# [0]: Prepare testing files
InputDirectory: str = 'model/test_benchmark'
OutputDirectory: str = 'model/inference_test_result'
FilePrefix: str = 'Testing Case #'
Files = {
    f'{FilePrefix}0': "Homolysis Dataset - Unique",
    f'{FilePrefix}5': "ALFABET's Comparison - No outlier",
    f'{FilePrefix}7': "DFT Dataset - Unique & No outlier",
    f'{FilePrefix}9': "Exp Dataset - Unique & No outlier",
    f'{FilePrefix}10': "Drug-Test 82",
    f'{FilePrefix}16': "Drug-Test 93 - Unique",
    f'{FilePrefix}18': "Methyl Linolenate",
}

FileParams = FileParseParams(mol=0, radical=(1, 2), bIdx=3, bType=4, target=5)
PredParams = PredictParams()
PredParams.average = True
PredParams.ensemble = True

# --------------------------------------------------------------------------------
# [1]: Data Initialization
dataset = FeatureData(trainable=False, retrainable=False)

# Placeholder must be initialized before the FeatureEngineer
SEED: int = 1
placeholder = DEPLOYMENT.MakeGroupPlaceholderV2(seed=SEED, name=None, verbose=True)
generator = FeatureEngineer(dataset=dataset)

tester: GroupTester = GroupTester(dataset=dataset, generator=generator, placeholder=placeholder)

# --------------------------------------------------------------------------------
# [3]: Predict all files
t = perf_counter()
count: int = 0
TIMING = {}
for InputFile, OutputFile in Files.items():
    print('-' * 120)
    print('InputFile:', InputFile)
    t1 = perf_counter()
    # [3.1]: Load the file
    InputFilePath = InputDirectory + f'/{InputFile}.csv'
    tester.AddOnDefinedFile(FilePath=InputFilePath, params=FileParams)

    # [3.2]: Running Feature Engineering
    tester.CreateData()

    # [3.3]: Doing the prediction
    result: pd.DataFrame = tester.predict(params=PredParams)
    # DisplayDataFrame(result, max_size=64)
    count += result.shape[0]

    # [3.4]: Export the File
    OutputFilePath = OutputDirectory + f'/S{SEED}' + f'/{OutputFile} - {placeholder.GetName()} [Pred].csv'
    if OutputFilePath is not None:
        # ExportFile(DataFrame=result, FilePath=OutputFilePath)
        pass
    TIMING[InputFile] = perf_counter() - t1
    print('-' * 120)

timer = perf_counter() - t
print(f'Total Time to Predict {len(Files)} with {count} reactions: {timer:.4f} (s) -> '
      f'{timer / count * 1000:.4f} (ms/reaction) or {count / timer:.4f} (reaction/s)')
print('Timing', TIMING)
