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
from Bit2Edge.test.SingleTester import SingleTester
from Bit2Edge.test.TesterUtilsP1 import DisplayDataFrame
from Bit2Edge.test.params.PredictParams import PredictParams
from Bit2Edge.utils.file_io import ExportFile

# --------------------------------------------------------------------------------
# [1]: Data Initialization
# You don't need to remember the :arg:`key` here as it have the default key here,
# The :arg:`key` here is made to test some customization

# You don't need to set up the placeholder as it is used for caching
# placeholder.Setup(dataset=dataset, key=None)
# Placeholder must be initialized before the FeatureEngineer
SEED: int = 0
RUN: int = 2
VER: int = 1
EPOCH: int = 79
DEPLOYMENT.BaseFilePath = 'resources/model/RGv1.6-BDE'
DEPLOYMENT.RunCode = 'Common_'
DEPLOYMENT.ModelTagSuffix = None
placeholder = DEPLOYMENT.MakeSinglePlaceholder(seed=SEED, run=RUN, ver=VER)


dataset = FeatureData(trainable=False, retrainable=False)
generator = FeatureEngineer(dataset=dataset)
tester: SingleTester = SingleTester(dataset=dataset, generator=generator, placeholder=placeholder)

# --------------------------------------------------------------------------------
# [2]: Predict a file
# FilePath: str = f'model/dataset/test/S{SEED}/Test-Dev Set [Pred].csv'
FilePath: str = f'model/dataset/test/need_test' + '/Testing Case #9.csv'
params = FileParseParams(mol=0, radical=(1, 2), bIdx=3, bType=4, target=5)
SORTING_FILE: bool = False
tester.AddOnDefinedFile(FilePath=FilePath, sorting=SORTING_FILE, params=params)

# [3]: Running Feature Engineering
tester.CreateData()

# [4]: Doing the prediction
PredParams = PredictParams()
PredParams.getLastLayer = True
result: pd.DataFrame = tester.predict(params=PredParams)
MAX_DISPLAY: int = 64
# DisplayDataFrame(result, max_size=MAX_DISPLAY)

ToFile: bool = True
# OutputFile: str = f'resources/TestCase [Pred] - {placeholder.GetName()}.csv'
OutputFile: str = 'resources/Testing Case #9 S0 [Pred-Ckpt].csv'
if OutputFile is not None and ToFile:
    ExportFile(DataFrame=result, FilePath=OutputFile)

# [5]: Doing visualization
VISUALIZATION: bool = False
if VISUALIZATION:
    vMode: str = 'lbi'
    model: str = 'UMAP'
    n_components: int = 2
    ref_mode: int = 0
    marker_trace: str = 'random'
    target_label_name = None
    bond_type_symbol = False
    detail_bond_type_symbol = True
    normalize_range = None
    dra_df = tester.Visualize(vMode=vMode, model=model, n_components=n_components, ref_mode=ref_mode,
                              marker_trace=marker_trace, target_label_name=target_label_name,
                              bond_type_symbol=bond_type_symbol, detail_bond_type_symbol=detail_bond_type_symbol,
                              normalize_range=normalize_range)

# [6]: Compute Feature Importance
FEATURE_IMPORTANCE: bool = False
if FEATURE_IMPORTANCE:
    method: int = 0
    n_runs: int = 10
    n_estimators: int = 30
    max_depth: int = 20
    feat_df = tester.FeatureImportance(method=method, n_runs=n_runs, n_estimators=n_estimators, max_depth=max_depth)
