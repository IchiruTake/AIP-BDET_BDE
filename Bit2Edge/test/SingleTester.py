# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This class served as a end-user exposed object to predict data given one model
# placeholder.
# --------------------------------------------------------------------------------
from logging import info, warning
from time import perf_counter, sleep
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray

from Bit2Edge.dataObject.DataBlock import DEFAULT_OUTPUT_NPDTYPE
from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.input.Creator import FeatureEngineer
from Bit2Edge.model.model import B2E_Model
from Bit2Edge.test.IntermediateTester import IntermediateTester
from Bit2Edge.test.params.PredictParams import PredictParams
from Bit2Edge.test.params.VisualizeEngine import VisualizeEngine
from Bit2Edge.utils.cleaning import RunGarbageCollection
from Bit2Edge.utils.verify import InputFullCheck, MeasureExecutionTime, TestState, TestStateByWarning

from Bit2Edge.test.placeholder.SinglePlaceholderV1 import SinglePlaceholderV1
from Bit2Edge.test.placeholder.SinglePlaceholderV2 import SinglePlaceholderV2

AcceptedPlaceholder = Union[SinglePlaceholderV1, SinglePlaceholderV2]


class SingleTester(IntermediateTester):

    @MeasureExecutionTime
    def __init__(self, dataset: FeatureData, generator: FeatureEngineer, placeholder: AcceptedPlaceholder,
                 GPU_MEM: bool = False, GPU_MASK: Tuple[bool, ...] = (True,)):
        # The automatic configuration must be loaded before initialization
        if not isinstance(placeholder, (SinglePlaceholderV1, SinglePlaceholderV2)):
            raise ValueError('The placeholder is not a single placeholder.')

        super(SingleTester, self).__init__(dataset=dataset, generator=generator, placeholder=placeholder,
                                           GPU_MEM=GPU_MEM, GPU_MASK=GPU_MASK)

    # [2]: Model's Prediction: ------------------------------------------------------------------------------------
    def predict(self, params: PredictParams) -> pd.DataFrame:
        """
        This method will predict the features in the `self.Dataset()` (Feature Data)

        Arguments:
        ---------

        params : PredictParams
            The parameters used for prediction
        
        Returns:
        -------

        A pandas.DataFrame
        """
        # [0]: Hyper-parameter Verification
        self._TestPredictRequirement_(force=params.force)
        InputFullCheck(params.getLastLayer, name='getLastLayer', dtype='bool')
        TestStateByWarning(self.GetDataInBlock(request='Info').shape[0] > int(3e5) and params.getLastLayer,
                           msg='Your data file contained too many samples it is hard to acquire all of those data.')
        sleep(1e-4)
        placeholder: AcceptedPlaceholder = self.GetPlaceholder()
        placeholder.SetupTFModel(dataset=self.Dataset(), key=None)

        # [1]: Predict target
        print('-' * 30, self.predict, '-' * 30)
        print('AIP-BDET is predicting data. Please wait for a secs ...')
        timer: float = perf_counter()
        self._CorePredict_(force=params.force, mode=params.mode, getLastLayer=params.getLastLayer,
                           verbose=params.verbose)
        self._UpdateTiming_(start_time=timer, dtype='predictFunction', timer_type=None)
        RunGarbageCollection(0)

        df: pd.DataFrame = super(SingleTester, self).ExportPredToDf(Sfs=params.Sfs)
        self._UpdateTiming_(start_time=timer, dtype='predictMethod', timer_type=None)
        self._DisplayPredTime_()
        return df

    def _CorePredict_(self, force: bool, mode: int, getLastLayer: bool, verbose: bool) -> None:
        # [1]: Check whether the data/feature is already completed
        EnvData: ndarray = self.GetDataInBlock(request='EnvData')
        LBIData: ndarray = self.GetDataInBlock(request='LBIData')
        DATA = (EnvData, LBIData)
        TF_MODEL: B2E_Model = self.GetPlaceholder().GetTFModel()

        # [2.1]: Calling the prediction function on Y_PRED
        ComputeCache = self.GetCache()
        if force or ComputeCache['prediction-data'] is None:
            ComputeCache['prediction-data'] = TF_MODEL.PredictV1(data=DATA, mode=mode, verbose=verbose)
            self._LabelCasting_()
        else:
            info('We already have the prediction >> Disable execution.')
            sleep(1e-4)

        # [2.3]: Calling the sub-model's prediction on Y_PRED
        if not getLastLayer:
            return None

        if force or ComputeCache['analysis-data'] is None:
            print('AIP-BDET is retrieved the last layer value: ...')
            ComputeCache['analysis-data'] = TF_MODEL.GetLastLayerOutput(data=DATA, mode=mode, verbose=verbose)
            ComputeCache['analysis-label'] = TF_MODEL.GetLastLayerName(verbose=verbose)
        else:
            info('We already have the result at the last layer >> Disable execution.')
            sleep(1e-4)

        return None

    # [5]: Other Methods ----------------------------------------------------------------------------------------------
    # [5.1]: Data Visualization ---------------------------------------------------------------------------------------
    def Visualize(self, engine: VisualizeEngine, vMode: str = 'lbi', ref_mode: Optional[int] = 0,
                  target_label_name: Optional[str] = None, bond_type_symbol: bool = True,
                  detail_bond_type_symbol: bool = True, normalize_range: Optional[Tuple] = None) -> pd.DataFrame:
        """
        This method will visualize the relationship between features and target. This method use the `plotly` package
        under the neath for data visualization.

        Arguments:
        ---------

        vMode : str
            The visualization model. If vMode='last', retrieve the last layer. If :arg:`vMode`='lbi', using
            the localized bond information (default).

        model : str
            The dimensionality reduction method. Default to be UMAP, model='UMAP'.

        n_components : int
            The number of features after dimensionality reduction, default to 2 but limited to 3.

        ref_mode : int
            The reference outcome which followed the column index of `self._compute_cache['performance-data']`.
            Default to :arg:`mode`=0.

        marker_trace : str
            The code for representation. Default to 'circle'.

        n_jobs : int
            Whether to use joblib parallelism in sklearn. Default to -1.

        args, kwargs: The optional argument for the model

        Returns:
        -------

        A pandas.DataFrame
        """
        # [0]: Hyper-parameter Verification & Pre-processing tasking
        timer: float = perf_counter()

        engine.evaluate()
        InputFullCheck(vMode, name='vMode', dtype='str')
        SELECTION = ('last', 'lbi')
        vMode = vMode.lower()
        TestState(vMode in SELECTION, f'Un-identified method. Please choose again: {SELECTION}.')

        TARGET, TARGET_LABEL, datatype = self._GetVisualReference_(ref_mode=ref_mode, preserve_dtype=True)
        if 'Target' in TARGET_LABEL:
            TARGET_LABEL = TARGET_LABEL.split('-')[1]
        if target_label_name is not None:
            print(f'The label name has switched from "{TARGET_LABEL}" to "{target_label_name}".')
            TARGET_LABEL = target_label_name

        print('-' * 35, self.Visualize, '-' * 35)
        print('[0]: Choose the data for visualization ...')

        import plotly.express as px

        print('[1]: Loading data for visualization ...')
        dra_time: float = perf_counter()
        if vMode == 'last':
            DATA, LABEL = self._VisualLastLayer_()
            n_components: int = DATA.shape[1]
        elif vMode == 'lbi':
            OPTION: str = 'LBI' if vMode == 'lbi' else 'Env'
            data: ndarray = self.GetDataInBlock(request=OPTION + 'Data')
            n_components: int = engine.n_components

            DATA = engine.Compute(data)
            LABEL = engine.GetVisualLabels()
        else:
            raise ValueError  # Never-reached but left there for code safety
        print(f'Dimensionality Reduction Time: {perf_counter() - dra_time:.4f} (s).')

        print('[2]: Prepare data for visualization ...')
        BOND_TYPE, BOND_TYPE_LABEL, _ = self._GetVisualReference_(ref_mode=-1)  # Bond-Type
        if bond_type_symbol and detail_bond_type_symbol:
            from Bit2Edge.molUtils.molUtils import DetermineDetailBondType
            p = self.GetParams()
            reactions = self.GetDataInBlock(request='Info')[:, [p.Mol(), p.BondIndex()]].tolist()
            for i, (smiles, bond_index) in enumerate(reactions):
                BOND_TYPE[i, 0] = DetermineDetailBondType(smiles=smiles, bondIdx=bond_index)

        df: pd.DataFrame = pd.DataFrame(data=BOND_TYPE, columns=[BOND_TYPE_LABEL], index=None)
        df[BOND_TYPE_LABEL] = df[BOND_TYPE_LABEL].astype(str)
        for idx, sub_label in enumerate(LABEL):
            temp = DATA[:, idx]
            if normalize_range is not None:
                temp = (normalize_range[1] - normalize_range[0]) * \
                       (temp - temp.min()) / (temp.max() - temp.min()) + \
                       normalize_range[0]
            df[sub_label] = temp
        df[TARGET_LABEL] = TARGET[:, 0]
        df.info()
        print(f'PLOTLY is drawing via the selection key ({vMode}). \n{df.head(5)}')

        print('[3]: Start to Visualize ...')
        COLOR = BOND_TYPE_LABEL if bond_type_symbol else TARGET_LABEL
        SYMBOL = BOND_TYPE_LABEL if bond_type_symbol else None
        OPACITY: float = 0.95
        if n_components == 1:
            FIGURE = px.scatter(df, x=LABEL[0], y=LABEL[1], color=COLOR, symbol=SYMBOL, opacity=OPACITY)
        elif n_components == 2 and vMode == 'lbi':
            FIGURE = px.scatter(df, x=LABEL[0], y=LABEL[1], color=COLOR, symbol=SYMBOL,
                                opacity=OPACITY)
        else:
            if n_components != 3:
                df[TARGET_LABEL] = df[TARGET_LABEL].astype(datatype)
            FIGURE = px.scatter_3d(df, x=LABEL[0], y=LABEL[1], z=LABEL[2] if n_components == 3 else TARGET_LABEL,
                                   color=COLOR, symbol=SYMBOL, opacity=OPACITY)

        if engine.marker_trace != 'random':
            FIGURE.update_traces(marker_symbol=engine.marker_trace)
        FIGURE.show(validate=False)

        self._UpdateTiming_(start_time=timer, dtype='visualize', timer_type='Data-Visualization')
        return df

    def _VisualLastLayer_(self) -> Tuple[ndarray, List[str]]:
        CACHE = self.GetCache()
        NUM_OUTPUT: int = len(CACHE['analysis-label'])
        if len(CACHE['analysis-label']) in (1, 2, 3):
            return CACHE['analysis-data'], CACHE['analysis-label']

        warning('The current AIP-BDET model and/or Bit2Edge structure may be incompatible.')
        OPTION: List[int] = []
        while len(set(OPTION)) not in (1, 2, 3):
            SELECTION: str = input(f'Choosing less or equal than three columns by INTEGER only '
                                   f'ranging from [0, {NUM_OUTPUT}] (separated by spacing): ')
            OUTPUTS = SELECTION.split()
            for out in OUTPUTS:
                if out.isdigit():
                    OPTION.append(int(out))
            if len(set(OPTION)) not in (1, 2, 3):
                OPTION.clear()

        return CACHE['analysis-data'][:, OPTION], [CACHE['analysis-label'][idx] for idx in OPTION]

    # [5.2]: Feature Importance ---------------------------------------------------------------------------------------
    @staticmethod
    def _GetEstimator_(method: int, n_estimators: int, max_depth: int, n_jobs: int = -1, **kwargs):
        from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, \
            RandomTreesEmbedding, ExtraTreesRegressor, BaggingRegressor
        from xgboost import XGBRegressor
        if method == 0:
            est = AdaBoostRegressor(n_estimators=n_estimators, **kwargs)
        elif method == 1:
            est = BaggingRegressor(n_estimators=n_estimators, n_jobs=n_jobs, **kwargs)
        elif method == 2:
            est = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, **kwargs)
        elif method == 3:
            est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, **kwargs)
        elif method == 4:
            est = RandomTreesEmbedding(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, **kwargs)
        elif method == 5:
            est = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, **kwargs)
        elif method == 6:
            est = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, **kwargs)
        else:
            raise ValueError
        print(est)
        return est

    def FeatureImportance(self, method: int, n_runs: int = 10, n_estimators: int = 30, max_depth: int = 20,
                          n_jobs: int = -1, **kwargs):
        print('-' * 30, 'Feature Importance', '-' * 30)
        CACHE = self.GetCache()
        DATA = CACHE['analysis-data']
        LABEL = CACHE['analysis-label']

        TestState(DATA is not None, msg='No analysis-data is computed, try predict with last layer.')
        TARGET, TARGET_LABEL, datatype = self._GetVisualReference_(ref_mode=0, preserve_dtype=True)

        # [1]: Get the estimator
        est = SingleTester._GetEstimator_(method=method, n_estimators=n_estimators, max_depth=max_depth,
                                          n_jobs=n_jobs, **kwargs)

        # [2]: Run the algorithm
        performance = np.zeros(shape=(n_runs, len(LABEL)), dtype=DEFAULT_OUTPUT_NPDTYPE)
        recorded_time = []
        for i in range(n_runs):
            print(f'Model Runtime: {int(i / n_runs * 100)} (%).')
            StartTime: float = perf_counter()
            try:
                est.fit(X=DATA, y=TARGET.ravel())
            except ValueError:
                est.fit(X=DATA, y=TARGET)
            performance[i, :] = est.feature_importances_
            recorded_time.append(perf_counter() - StartTime)

        # [3]: Report the performance
        mean = np.mean(recorded_time, dtype=DEFAULT_OUTPUT_NPDTYPE)
        std = np.std(recorded_time, dtype=DEFAULT_OUTPUT_NPDTYPE)
        print(f'Average Time: {mean:.4f} ± {std:.4f} (s).')
        for idx, label in enumerate(LABEL):
            data = performance[:, idx]
            mean, std = np.mean(data, dtype=DEFAULT_OUTPUT_NPDTYPE), np.std(data, dtype=DEFAULT_OUTPUT_NPDTYPE)
            print(f'Label: {label} --> Importance: {mean * 100:.4f} ± {std * 100:.4f} (%)')

        return pd.DataFrame(data=performance, columns=LABEL)

