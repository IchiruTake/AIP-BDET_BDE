# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This class served as a general-purpose, intermediate object to perform the
# target mapping in a correct order
# --------------------------------------------------------------------------------

from typing import List, Optional

import numpy as np
from numpy import ndarray

from Bit2Edge.config.modelConfig import MODEL_STRUCTURE
from Bit2Edge.dataObject.DataBlock import DEFAULT_OUTPUT_NPDTYPE
from Bit2Edge.utils.verify import (InputFullCheck, TestState)


class TargetDefinition:
    # -----------------------------------------------------------------------------------
    # Label processor
    @staticmethod
    def ComputeLabel(index: int, notion: str, term: str) -> str:
        return f'#{index}-{notion}: {term}'

    @staticmethod
    def DecodeLabel(label: str) -> dict:
        """
        Normally, our label is in the format of `#{index}-{notion}: {term}`, which can be found in the
        :meth:`TargetDefinition.ComputeLabel` method. This method is used to decode the label into a dictionary.

        This is mostly used in the prediction code: The name is '#0-BDE: pred' for prediction and '#0-BDE: ref' for
        reference.

        """
        index, code = label.split('-')
        notion, term = code.split(': ')
        TestState(label == TargetDefinition.ComputeLabel(index=int(index[1:]), notion=notion, term=term),
                  msg='The label is not in the correct format.')
        return {'index': index[1:], 'notion': notion, 'term': term}

    @staticmethod
    def GetTemplatePerfColumnForOneOutput() -> List[str]:
        return ['Target', 'Predict', 'RelError', 'AbsError']

    # -----------------------------------------------------------------------------------
    @staticmethod
    def MapTarget(prediction: ndarray, target: Optional[ndarray], num_pred: int) -> dict:
        InputFullCheck(prediction, name='prediction', dtype='ndarray')
        InputFullCheck(target, name='target', dtype='ndarray-None', delimiter='-')
        InputFullCheck(num_pred, name='num_pred', dtype='int')
        result = {}

        # [1]: Ensure the target is correct and compatible with definition
        PREDICTION_SIZE, PREDICTION_REMAINDER = divmod(prediction.shape[1], num_pred)
        print({'pred_shape': prediction.shape, 'num_join_pred': num_pred,
               'pred_size': PREDICTION_SIZE, 'pred_remainder': PREDICTION_REMAINDER})
        TestState(PREDICTION_REMAINDER == 0, msg='The prediction cache is wrongly computed.')
        if target is not None:
            TestState(target.shape[1] == PREDICTION_SIZE,
                      msg='The provided target size is incompatible with the model prediction.')

        # [2]: Create target label
        result['prediction-label'] = TargetDefinition.GetPredictionLabels(num_output_each=PREDICTION_SIZE,
                                                                          num_model=num_pred)
        result['performance-label'] = TargetDefinition.GetPerformanceLabels(num_output_each=PREDICTION_SIZE,
                                                                            num_model=num_pred)

        # [3]: Create performance data
        template = TargetDefinition.GetTemplatePerfColumnForOneOutput()
        temp = np.zeros(shape=(prediction.shape[0], len(result['performance-label'])),
                        dtype=prediction.dtype or DEFAULT_OUTPUT_NPDTYPE)
        result['performance-data'] = temp
        for i in range(0, prediction.shape[1]):
            ptr: int = len(template) * i
            temp[:, ptr + 1] = prediction[:, i]
            if target is not None:
                # Allow partial target but must be ordering equivalently --> NOT Recommended
                temp[:, ptr + 0] = target[:, i % target.shape[1]]
                temp[:, ptr + 2] = temp[:, ptr + 0] - temp[:, ptr + 1]
                temp[:, ptr + 3] = np.absolute(temp[:, ptr + 2])

        return result

    @staticmethod
    def GetDefinedPredictionNotion(num_output: int) -> List[str]:
        NOTIONS = MODEL_STRUCTURE.get('Predict-Notion', None)
        TestState(NOTIONS is not None, 'The prediction notion is not found, please check the config file.')
        TestState(len(NOTIONS) == len(set(NOTIONS)),
                  msg='The prediction notion is not unique, please check the config file.')
        if num_output == 1:
            idx = MODEL_STRUCTURE.get('Predict-Notion-Index', 0)
            notions = [NOTIONS[idx], ]  # 'a' notions -> len(GetTemplatePerfColumnForOneOutput) *'a' columns
        else:
            TestState(num_output <= len(NOTIONS), 'The number of output is exceeded the model representation.')
            notions = NOTIONS[:num_output]
        return notions

    @staticmethod
    def GetPerformanceLabels(num_output_each: int, num_model: int) -> List[str]:
        NOTIONS = TargetDefinition.GetDefinedPredictionNotion(num_output=num_output_each)

        TemplateLabel: List[str] = TargetDefinition.GetTemplatePerfColumnForOneOutput()
        TemplateLabelSize: int = len(TemplateLabel)

        FullLabels: List[str] = list(TemplateLabel) * num_output_each * num_model
        Counter: int = 0
        # For two nested-loop, index is `notion_idx * TemplateLabelSize + label_idx`
        # For three nested-loop, index is `model_idx * notion_idx * TemplateLabelSize + notion_idx * label_idx`
        # This resulted in an over-whelming complex output labels
        for model_idx in range(num_model):
            for notion_idx, notion in enumerate(NOTIONS):
                for label_idx in range(0, TemplateLabelSize):
                    value: str = FullLabels[Counter]
                    FullLabels[Counter] = TargetDefinition.ComputeLabel(index=model_idx, notion=notion, term=value)
                    Counter += 1
        return FullLabels

    @staticmethod
    def GetPredictionLabels(num_output_each: int, num_model: int, term: str = 'pred') -> List[str]:
        NOTIONS = TargetDefinition.GetDefinedPredictionNotion(num_output=num_output_each)
        FullLabels: List[str] = list(NOTIONS) * num_model
        Counter: int = 0
        # For two nested-loop, index is `notion_idx * TemplateLabelSize + label_idx`
        # For three nested-loop, index is `model_idx * notion_idx * TemplateLabelSize + notion_idx * label_idx`
        # This resulted in an over-whelming complex output labels
        for model_idx in range(num_model):
            for _, notion in enumerate(NOTIONS):
                FullLabels[Counter] = TargetDefinition.ComputeLabel(index=model_idx, notion=notion, term=term)
                Counter += 1
        return FullLabels


