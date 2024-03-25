# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This class served as a general-purpose, intermediate object to store all models
# that have the same input structure before and after CleanEnvData() function.
#
# --------------------------------------------------------------------------------

from logging import warning
from typing import Union, Dict, Tuple, Optional, List

from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.test.placeholder.SinglePlaceholderV2 import SinglePlaceholderV2
from Bit2Edge.utils.verify import TestState, TestStateByWarning


class GroupPlaceholder:
    MAXSIZE: int = 30

    def __init__(self, name: Optional[str] = None):
        super(GroupPlaceholder, self).__init__()
        self._name: str = name
        self._placeholders: Dict[str, SinglePlaceholderV2] = {}
        self._weights: Optional[Tuple[float, ...]] = None

    def SetWeights(self, value: Union[List[float], Tuple[float, ...]]) -> None:
        self._weights = tuple(value)

    def GetWeights(self) -> Optional[Tuple[float, ...]]:
        return self._weights

    def GetWeightsSafely(self, num_output: int) -> Tuple[float, ...]:
        weights = self.GetWeights()
        if weights is None or not weights:
            return tuple([0.0] * num_output)
        return weights

    def GetName(self) -> str:
        return self._name

    def GetPlaceholders(self) -> Dict[str, SinglePlaceholderV2]:
        return self._placeholders

    def GetPlaceholder(self, name: Union[int, str]) -> SinglePlaceholderV2:
        return self._placeholders[name]

    def GetNumModels(self) -> int:
        return len(self._placeholders)

    def __len__(self) -> int:
        return self.GetNumModels()

    def __getitem__(self, item: int) -> SinglePlaceholderV2:
        return self.GetPlaceholder(name=item)

    GetPlaceholderListSize = GetNumModels

    # --------------------------------------------------------------------------------
    def Register(self, placeholder: Optional[SinglePlaceholderV2]) -> None:
        if not isinstance(placeholder, SinglePlaceholderV2):
            raise TypeError(f'Incorrect comparison between {self.__class__} and {type(placeholder)}.')

        TestStateByWarning(len(self._placeholders) < GroupPlaceholder.MAXSIZE,
                           f'The number of placeholders is exceeded {GroupPlaceholder.MAXSIZE} units.')
        placeholder_name: str = placeholder.GetName()
        if placeholder_name not in self._placeholders:
            self._placeholders[placeholder_name] = placeholder
        elif placeholder_name in self._placeholders:
            warning(f'The placeholder with name={placeholder_name} is already registered.')
        return None

    def Unregister(self, name: str) -> Optional[SinglePlaceholderV2]:
        if self._placeholders and name in self._placeholders:
            return self._placeholders.pop(name, None)

        warning(f'The group did not receive any placeholder that has name {name}.')
        return None

    def SetupTFModels(self, dataset: FeatureData, key: Optional[str] = None) -> None:
        for i, (_, placeholder) in enumerate(self._placeholders.items()):
            if placeholder.GetTFModel() is None:
                placeholder.SetupTFModel(dataset=dataset, key=key)
            else:
                TestState(dataset is placeholder.GetTFModel().Dataset(),
                          msg=f'The FeatureData defined in this placeholder (idx={i}) is not equivalent.')
        return None
