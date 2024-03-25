# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------
# This class is served to be a pool of loaded TensorFlow Model.
# Since we know that deleting a model will not release the memory, we need to
# keep the model in memory and reuse it, especially the inference of multi-model.
# --------------------------------------------------------------------------------

import filecmp
from logging import warning
from os.path import isfile
from typing import Dict, Optional

# From v2.6, Tensorflow.Keras is a directory passed to Keras library.
from tensorflow.keras.models import Model, load_model


class ModelPool:
    POOL: Dict[str, Model] = {}

    @staticmethod
    def GetModel(filepath: str, compilation: bool = False, **kwargs) -> Model:
        result = ModelPool._GetModelInternal_(filepath=filepath)
        if result is not None:
            return result
        warning(f'Model {filepath} is not in the pool. Loaded an item to the pool.')
        result = load_model(filepath=filepath, compile=compilation, **kwargs)
        ModelPool.POOL[filepath] = result
        return result

    @staticmethod
    def _GetModelInternal_(filepath: str) -> Optional[Model]:
        if not ModelPool.IsValidFilePath(filepath):
            return None
        if filepath in ModelPool.POOL:
            return ModelPool.POOL[filepath]
        for name, model in ModelPool.POOL.items():
            if filecmp.cmp(name, filepath, shallow=True):
                return model
        return None

    @staticmethod
    def IsModelInPool(filepath: str) -> bool:
        return ModelPool._GetModelInternal_(filepath) is not None

    @staticmethod
    def IsValidFilePath(filepath: str) -> bool:
        return isfile(filepath)

    @staticmethod
    def ClearFileCmpCache():
        filecmp.clear_cache()

    @staticmethod
    def ClearPool():
        ModelPool.POOL.clear()
        ModelPool.ClearFileCmpCache()
