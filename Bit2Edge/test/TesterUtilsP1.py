# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from sys import maxsize
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import (NMF, PCA, DictionaryLearning, FactorAnalysis, IncrementalPCA,
                                   KernelPCA, LatentDirichletAllocation, MiniBatchDictionaryLearning,
                                   MiniBatchSparsePCA, SparsePCA, TruncatedSVD)
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap, LocallyLinearEmbedding, MDS

from Bit2Edge.dataObject.Dataset import FeatureData
from Bit2Edge.input.Creator import FeatureEngineer
from Bit2Edge.utils.verify import TestState

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 300)
np.set_printoptions(threshold=maxsize)


# [1]: Configure the BaseTester ----------------------------------------------------------------------------------
def ConfigureEngineForTester(dataset: FeatureData, generator: FeatureEngineer) -> Tuple[FeatureData, FeatureEngineer]:
    TestState(dataset.OptFeatureSetKey() == 'Test', 'This object::dataset is in-valid.')
    if generator is None:
        generator = FeatureEngineer(dataset=dataset, storeEnvironment=False, showInvalidStereo=False, TIMER=False)
    TestState(generator.Dataset() is dataset, 'The generator must be linked to the same object::dataset.')
    return dataset, generator


def EnableGPUDevice(mask: Union[List[bool], Tuple[bool, ...]] = (True,)) -> None:
    from tensorflow import config
    dev = config.list_physical_devices('GPU')
    for idx, state in enumerate(mask):
        config.experimental.set_memory_growth(dev[idx], state)
    return None


def DisplayDataFrame(df: pd.DataFrame, max_size: int = 32) -> None:
    print('-' * 35, 'Display', '-' * 36)
    print(df.head(min(df.shape[0], max_size)))
    print('-' * 81)


def CastTarget(value: float, Sfs: int) -> str:
    return ("{0:0." + str(Sfs) + "f}").format(value)


# [2]: Dimensionality-Reduction Analysis ---------------------------------------------------------------------------
def BuildVisualModel(model: str, n_components: int, n_jobs: int, *args, **kwargs) \
        -> Union[TransformerMixin, BaseEstimator]:
    _NUM_NEIGHBORS: int = 50
    _LEARNING_RATE: float = 0.75

    if model == 'pca':
        return PCA(n_components=n_components, n_jobs=n_jobs, *args, **kwargs)
    if model == 'k-pca':
        return KernelPCA(n_components=n_components, *args, **kwargs)
    if model == 's-pca':
        return SparsePCA(n_components=n_components, verbose=1, *args, **kwargs)
    if model == 'mini s-pca':
        return MiniBatchSparsePCA(n_components=n_components, n_jobs=n_jobs, *args, **kwargs)
    if model == 'i-pca':
        return IncrementalPCA(n_components=n_components)
    if model == 't-svd':
        return TruncatedSVD(n_components=n_components)
    if model == 'lda':
        return LatentDirichletAllocation(n_components=n_components, n_jobs=n_jobs, *args, **kwargs)
    if model == 'fa':
        return FactorAnalysis(n_components=n_components, *args, **kwargs)
    if model == 'mini dict-learn':
        return MiniBatchDictionaryLearning(n_components=n_components, n_jobs=n_jobs, *args, **kwargs)
    if model == 'dict-learn':
        return DictionaryLearning(n_components=n_components, n_jobs=n_jobs, *args, **kwargs)
    if model == 'nmf':
        return NMF(n_components=n_components, *args, **kwargs)
    if model == 'isomap':
        return Isomap(n_neighbors=_NUM_NEIGHBORS, n_components=n_components,
                      n_jobs=n_jobs, *args, **kwargs)
    if model == 'se':
        return SpectralEmbedding(n_neighbors=_NUM_NEIGHBORS, n_components=n_components,
                                 n_jobs=n_jobs, *args, **kwargs)
    if model == 'lle':
        return LocallyLinearEmbedding(n_neighbors=_NUM_NEIGHBORS, n_components=n_components,
                                      n_jobs=n_jobs, *args, **kwargs)
    if model == 'mds':
        return MDS(n_neighbors=_NUM_NEIGHBORS, n_components=n_components,
                   verbose=1, n_jobs=n_jobs, *args, **kwargs)
    if model == 'tsne':
        PERPLEXITY: int = 50
        return TSNE(n_components=n_components, perplexity=PERPLEXITY,
                    learning_rate=_LEARNING_RATE, n_jobs=n_jobs, *args, **kwargs)
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError('UMAP is not installed. Please install it first.')
    print('Default to Uniform Manifold Approximation and Projection (UMAP).')
    _N_EPOCHS: int = 300
    _MIN_DIST: float = 0.25
    _LOW_MEMORY: bool = True
    _VERBOSE: bool = True
    _UNIQUE: bool = True
    return UMAP(n_neighbors=_NUM_NEIGHBORS, n_components=n_components, n_epochs=_N_EPOCHS, learning_rate=_LEARNING_RATE,
                low_memory=_LOW_MEMORY, min_dist=_MIN_DIST, verbose=_VERBOSE, unique=_UNIQUE, n_jobs=n_jobs,
                *args, **kwargs)


def ComputeDRA(data: ndarray, model: str, n_components: int, n_jobs: int = -1, *args, **kwargs) -> ndarray:
    MODEL = BuildVisualModel(model, n_components, n_jobs, *args, **kwargs)
    TestState(hasattr(MODEL, 'fit_transform'), 'The method `fit_transform()` must be found in the SkLearn model.')
    return MODEL.fit_transform(data)
