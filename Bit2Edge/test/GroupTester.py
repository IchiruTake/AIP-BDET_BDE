# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This class served as a end-user exposed object to predict data given one group
# placeholder.
# --------------------------------------------------------------------------------

from logging import info
from time import perf_counter, sleep
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from numpy import ndarray

from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.input.Creator import FeatureEngineer
from Bit2Edge.model.model import B2E_Model
from Bit2Edge.test.IntermediateTester import IntermediateTester
from Bit2Edge.test.params.PredictParams import PredictParams
from Bit2Edge.utils.cleaning import RunGarbageCollection
from Bit2Edge.utils.verify import MeasureExecutionTime, TestState
from Bit2Edge.test.placeholder.GroupPlaceholder import GroupPlaceholder

AcceptedPlaceholder = Union[GroupPlaceholder]


class GroupTester(IntermediateTester):

    @MeasureExecutionTime
    def __init__(self, dataset: FeatureData, generator: FeatureEngineer, placeholder: AcceptedPlaceholder,
                 GPU_MEM: bool = False, GPU_MASK: Tuple[bool, ...] = (True,)):
        # The automatic configuration must be loaded before initialization
        if not isinstance(placeholder, GroupPlaceholder):
            raise ValueError('The :var:`placeholder` must be an instance of GroupPlaceholderV1 or GroupPlaceholderV2.')

        super(GroupTester, self).__init__(dataset=dataset, generator=generator, placeholder=placeholder,
                                          GPU_MEM=GPU_MEM, GPU_MASK=GPU_MASK)
        CACHE = self.GetCache()
        CACHE['average'] = False
        CACHE['average-data'] = None
        CACHE['ensemble'] = False
        CACHE['ensemble-data'] = None

    # [2]: Model's Prediction: ------------------------------------------------------------------------------------
    def predict(self, params: Optional[PredictParams] = None) -> pd.DataFrame:
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
        if params is None:
            params = PredictParams()

        self._TestPredictRequirement_(force=params.force)
        placeholder: AcceptedPlaceholder = self.GetPlaceholder()
        placeholder.SetupTFModels(dataset=self.Dataset())

        # [1]: Predict target
        print('-' * 30, self.predict, '-' * 30)
        print('AIP-BDET is predicting data. Please wait for a secs ...')
        timer: float = perf_counter()
        self._CorePredict_(params=params)
        self._UpdateTiming_(start_time=timer, dtype='predictFunction', timer_type=None)
        RunGarbageCollection(1)

        # [2]: Compute Target
        df: Optional[pd.DataFrame] = self.ExportPredToDf(Sfs=params.Sfs)
        self._UpdateTiming_(start_time=timer, dtype='predictMethod', timer_type=None)
        self._DisplayPredTime_()
        return df

    def _CorePredict_(self, params: PredictParams) -> None:
        # [1]: Check whether the data/feature is already completed
        EnvData: ndarray = self.GetDataInBlock(request='EnvData')
        LBIData: ndarray = self.GetDataInBlock(request='LBIData')

        ToTensor: bool = True  # LBIData.shape[0] < 50000
        DATA = (EnvData, LBIData)

        # [2.1]: Calling the prediction function on Y_PRED
        PREDICTION: List[ndarray] = []
        NUM_OUTPUTS: List[int] = []  # For tracking purpose
        PLACEHOLDER: AcceptedPlaceholder = self.GetPlaceholder()
        INDEX_CACHE: Dict[int, str] = {}
        NUM_MODELS: int = len(PLACEHOLDER)
        ComputeCache = self.GetCache()

        if not params.force and ComputeCache['prediction-data'] is not None:
            info('We already have the prediction >> Disable execution.')
            sleep(1e-4)
            return None

        F_DATA, R_DATA = None, None
        for idx, (name, placeholder) in enumerate(PLACEHOLDER.GetPlaceholders().items()):
            model: B2E_Model = placeholder.GetTFModel()
            if F_DATA is None and R_DATA is None:  # and ShouldReuse
                F_DATA, R_DATA = model.GetDataV2(data=DATA, mode=params.mode, toTensor=ToTensor)
            result = model.PredictV2(fdata=F_DATA, rdata=R_DATA, mode=params.mode, verbose=params.verbose)
            RunGarbageCollection(0)

            PREDICTION.append(result)
            NUM_OUTPUTS.append(result.shape[-1])
            INDEX_CACHE[idx] = name

        if isinstance(F_DATA, List):
            F_DATA.clear()
        if isinstance(R_DATA, List):
            R_DATA.clear()
        del F_DATA, R_DATA
        RunGarbageCollection()

        # https://www.geeksforgeeks.org/python-convert-a-list-into-a-tuple/
        # Tuple-unpacking: Same as key unpacking in python dictionary
        TestState(len(set(NUM_OUTPUTS)) == 1,
                  msg=f'Some models in the placeholder (={NUM_OUTPUTS}) is not equivalent in size.')

        # Performing result-averaging
        timer: float = perf_counter()
        ComputeCache['average'] = params.average
        if params.average:
            AverageResult: ndarray = sum(PREDICTION) / NUM_MODELS
            PREDICTION.append(AverageResult)
            NUM_OUTPUTS.append(AverageResult.shape[-1])
            ComputeCache['average-data'] = AverageResult

        # Performing result-ensemble
        ComputeCache['ensemble'] = params.ensemble
        if params.ensemble:
            EnsembleResult: ndarray = np.zeros(shape=PREDICTION[-1].shape, dtype=PREDICTION[-1].dtype)
            for idx, key in INDEX_CACHE.items():
                weights = PLACEHOLDER.GetPlaceholder(name=key).GetWeights()
                TestState(weights is not None and len(weights) != 0,
                          msg='The :var:`weights` must be configured before.')
                EnsembleResult += np.multiply(PREDICTION[idx], weights)
            EnsembleResult += PLACEHOLDER.GetWeightsSafely(num_output=NUM_OUTPUTS[-1])

            PREDICTION.append(EnsembleResult)
            NUM_OUTPUTS.append(EnsembleResult.shape[-1])
            ComputeCache['ensemble-data'] = EnsembleResult

        if params.average or params.ensemble:
            print(f'Average ({params.average}) & Ensemble ({params.ensemble}) Time: {perf_counter() - timer:.4f} (s).')

        ComputeCache['prediction-data'] = np.concatenate((*PREDICTION,), axis=1)
        return self._LabelCasting_()
