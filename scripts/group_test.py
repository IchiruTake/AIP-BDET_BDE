# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
import pandas as pd

from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.dataObject.FileParseParams import FileParseParams
from Bit2Edge.deploy.deploy import DEPLOYMENT
from Bit2Edge.input.Creator import FeatureEngineer
from Bit2Edge.test.GroupTester import GroupTester
from Bit2Edge.test.TesterUtilsP1 import DisplayDataFrame
from Bit2Edge.test.params.PredictParams import PredictParams
from Bit2Edge.utils.file_io import ExportFile

# --------------------------------------------------------------------------------
# [1]: Data Initialization
dataset = FeatureData(trainable=False, retrainable=False)

# Placeholder must be initialized before the FeatureEngineer
SEED: int = 1
GROUP_VERSION: int = 1
if GROUP_VERSION == 1:
    placeholder = DEPLOYMENT.MakeGroupPlaceholderV1(seed=SEED, name=None, verbose=True)
elif GROUP_VERSION == 2:
    placeholder = DEPLOYMENT.MakeGroupPlaceholderV2(seed=SEED, name=None, verbose=True)
else:
    raise ValueError('Unsupported group version')
generator = FeatureEngineer(dataset=dataset)

tester: GroupTester = GroupTester(dataset=dataset, generator=generator, placeholder=placeholder)

# --------------------------------------------------------------------------------
# [2]: Predict a file
FilePath: str = 'resources/TestCase.csv'
params = FileParseParams(mol=0, radical=(1, 2), bIdx=3, bType=4, target=5)
tester.AddOnDefinedFile(FilePath=FilePath, params=params)

# [3]: Running Feature Engineering
tester.CreateData()

# [4]: Doing the prediction
PredParams = PredictParams()
result: pd.DataFrame = tester.predict(params=PredParams)
MAX_DISPLAY: int = 64
DisplayDataFrame(result, max_size=MAX_DISPLAY)
OutputFile: str = f'resources/TestCase [Pred] - {placeholder.GetName()}.csv'
if OutputFile is not None:
    ExportFile(DataFrame=result, FilePath=OutputFile)
