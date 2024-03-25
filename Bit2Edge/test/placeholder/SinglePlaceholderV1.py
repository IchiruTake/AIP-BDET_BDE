# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
from logging import warning
from typing import Dict, Optional

from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.model.model import B2E_Model
from Bit2Edge.test.placeholder.BasePlaceholder import BasePlaceholder

class SinglePlaceholderV1(BasePlaceholder):

    def __init__(self, ModelConfig: Dict[str, str], name: str = None):
        super(SinglePlaceholderV1, self).__init__(ModelConfig=ModelConfig, name=name)
        self._TF_Model: Optional[B2E_Model] = None

    # --------------------------------------------------------------------------------
    def SetupTFModel(self, dataset: FeatureData, key: Optional[str] = None) -> None:
        if self._TF_Model is None:
            self._TF_Model: B2E_Model = B2E_Model(dataset=dataset, TF_Model=self.GetTFModelPath(),
                                                  ModelKey=key or self.GetModelKey())
        elif dataset is not self._TF_Model.Dataset():
            self._TF_Model.BindNewDataset(dataset=dataset)
        else:
            warning('The new :arg:`dataset` is already configured inside the B2E_Model.')
        return None

    def Setup(self, dataset: Optional[FeatureData] = None, key: Optional[str] = None) -> None:
        if dataset is not None:
            self.SetupTFModel(dataset=dataset, key=key)
        super(BasePlaceholder, self).Setup()

    def GetTFModel(self) -> B2E_Model:
        return self._TF_Model

    # --------------------------------------------------------------------------------
    @staticmethod
    def GetNumModels() -> int:
        return 1

    def GetNumDefinedModels(self) -> int:
        return int(self.GetTFModel() is not None)

    def __len__(self) -> int:
        return self.GetNumModels()
