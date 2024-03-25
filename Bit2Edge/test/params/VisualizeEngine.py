# --------------------------------------------------------------------------------
#  Copyright (C) 2022 by Ichiru Take
#   @@ All Rights Reserved @@
#
#  This file is part of the Bit2EdgeV2-BDE program.
#  The contents are covered by the terms of the MIT license which is included
#  in the file license.txt, found at the root of the Bit2EdgeV2-BDE source tree.
# --------------------------------------------------------------------------------

from logging import info
from typing import Optional, Union, List

from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.decomposition import (NMF, PCA, DictionaryLearning, FactorAnalysis, IncrementalPCA,
                                   KernelPCA, LatentDirichletAllocation, MiniBatchDictionaryLearning,
                                   MiniBatchSparsePCA, SparsePCA, TruncatedSVD)
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap, LocallyLinearEmbedding, MDS

from Bit2Edge.dataObject.InputState import InputState
from Bit2Edge.utils.verify import InputCheckRange, InputFullCheck, TestState


class VisualizeEngine:
    __slots__ = ('model_name', 'n_components', 'marker_trace', 'n_jobs', 'params', 'model',
                 'default_config')

    def __init__(self, model_name: str = 'UMAP', n_components: int = 2, marker_trace: str = 'circle',
                 n_jobs: int = -1, **kwargs):
        self.model_name: str = model_name
        self.n_components: int = n_components
        self.marker_trace: str = marker_trace
        self.n_jobs: int = n_jobs
        self.params: dict = kwargs
        self.default_config: dict = {
            'NUM_NEIGHBORS': 50,
            'LEARNING_RATE': 0.75,
            'N_EPOCHS': 300,
            'MIN_DIST': 0.25,
            'LOW_MEM': True,
            'VERBOSE': True,
            'UNIQUE': True
        }
        self.model: Optional[Union[TransformerMixin, BaseEstimator]] = None
        self.evaluate()
    
    def evaluate(self) -> None:
        # [1]: Check marker_trace
        InputFullCheck(self.marker_trace, name='marker_trace', dtype='str')
        self.marker_trace = self.marker_trace.lower()
        ACCEPTED_MARKERS = ('random', 'circle', 'circle-open', 'square', 'square-open', 
                            'diamond', 'diamond-open', 'cross', 'x')
        if self.marker_trace not in ACCEPTED_MARKERS:
            link: str = 'https://plotly.com/python/reference/scattergl/#scattergl-marker-symbol'
            MARKER_TRACE: str = 'circle'
            info(f'See all plotly markers at here: {link}. Switch back to default (={MARKER_TRACE}')

        # [2]: Check model and other properties
        InputFullCheck(self.model_name, name='model_name', dtype='str')
        InputCheckRange(self.n_components, name='n_components', maxValue=3, minValue=1, rightBound=True)
        InputFullCheck(self.n_jobs, name='n_jobs', dtype='int')

        return None

    def GetDefaultConf(self, name: str) -> Optional[Union[int, bool]]:
        return self.default_config.get(name, None)

    def GenModel(self) -> Union[TransformerMixin, BaseEstimator]:
        _NUM_NEIGHBORS: int = self.GetDefaultConf('NUM_NEIGHBORS')
        _LEARNING_RATE: float = self.GetDefaultConf('LEARNING_RATE')
        _N_EPOCHS: int = self.GetDefaultConf('N_EPOCHS')
        _MIN_DIST: float = self.GetDefaultConf('MIN_DIST')
        _LOW_MEMORY: bool = self.GetDefaultConf('LOW_MEM')
        _VERBOSE: bool = self.GetDefaultConf('VERBOSE')
        _UNIQUE: bool = self.GetDefaultConf('UNIQUE')

        if self.model_name == 'pca':
            return PCA(n_components=self.n_components, n_jobs=self.n_jobs, **self.params)
        if self.model_name == 'k-pca':
            return KernelPCA(n_components=self.n_components, **self.params)
        if self.model_name == 's-pca':
            return SparsePCA(n_components=self.n_components, verbose=1, **self.params)
        if self.model_name == 'mini s-pca':
            return MiniBatchSparsePCA(n_components=self.n_components, n_jobs=self.n_jobs, **self.params)
        if self.model_name == 'i-pca':
            return IncrementalPCA(n_components=self.n_components)
        if self.model_name == 't-svd':
            return TruncatedSVD(n_components=self.n_components)
        if self.model_name == 'lda':
            return LatentDirichletAllocation(n_components=self.n_components, n_jobs=self.n_jobs, **self.params)
        if self.model_name == 'fa':
            return FactorAnalysis(n_components=self.n_components, **self.params)
        if self.model_name == 'mini dict-learn':
            return MiniBatchDictionaryLearning(n_components=self.n_components, n_jobs=self.n_jobs, **self.params)
        if self.model_name == 'dict-learn':
            return DictionaryLearning(n_components=self.n_components, n_jobs=self.n_jobs, **self.params)
        if self.model_name == 'nmf':
            return NMF(n_components=self.n_components, **self.params)
        if self.model_name == 'isomap':
            return Isomap(n_neighbors=_NUM_NEIGHBORS, n_components=self.n_components,
                          n_jobs=self.n_jobs, **self.params)
        if self.model_name == 'se':
            return SpectralEmbedding(n_neighbors=_NUM_NEIGHBORS, n_components=self.n_components,
                                     n_jobs=self.n_jobs, **self.params)
        if self.model_name == 'lle':
            return LocallyLinearEmbedding(n_neighbors=_NUM_NEIGHBORS, n_components=self.n_components,
                                          n_jobs=self.n_jobs, **self.params)
        if self.model_name == 'mds':
            return MDS(n_neighbors=_NUM_NEIGHBORS, n_components=self.n_components,
                       verbose=1, n_jobs=self.n_jobs, **self.params)
        if self.model_name == 'tsne':
            PERPLEXITY: int = 50
            return TSNE(n_components=self.n_components, perplexity=PERPLEXITY,
                        learning_rate=_LEARNING_RATE, n_jobs=self.n_jobs, **self.params)
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError('UMAP is not installed. Please install it first.')
        print('Default to Uniform Manifold Approximation and Projection (UMAP).')

        return UMAP(n_neighbors=_NUM_NEIGHBORS, n_components=self.n_components, n_epochs=_N_EPOCHS,
                    learning_rate=_LEARNING_RATE, low_memory=_LOW_MEMORY, min_dist=_MIN_DIST, 
                    verbose=_VERBOSE, unique=_UNIQUE, n_jobs=self.n_jobs,
                    **self.params)

    def GetModel(self):
        if self.model is None:
            self.model = self.GenModel()
        return self.model

    def Compute(self, data: ndarray) -> ndarray:
        model = self.GetModel()
        TestState(hasattr(model, 'fit_transform'), msg='The method `fit_transform()` is not found.')
        return model.fit_transform(data)

    @staticmethod
    def GetVisualLabels(self) -> List[str]:
        return [f'{unit}-axis' for unit in InputState.GetFullNames()[0:self.n_components]]
