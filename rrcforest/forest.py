import sys
from typing import Optional, Sequence, Mapping, Deque, Iterable, Generator
import itertools
from collections import deque

import pandas as pd
from rrcf import rrcf
from sklearn import preprocessing
from tqdm import tqdm
import numpy as np


class LagFeatures:
    _buffer: Deque[Mapping[str, float]]
    _columns: Sequence[str]

    def __init__(self, n_lags: int, columns: Sequence[str]):
        self._columns = columns
        self._buffer = deque(
            n_lags * [{col: np.nan for col in columns}],
            n_lags,
        )

    @property
    def n_lags(self) -> int:
        assert self._buffer.maxlen is not None
        return self._buffer.maxlen

    @property
    def feature_columns(self) -> Sequence[str]:
        return list(self.features.keys())

    @property
    def columns(self) -> Sequence[str]:
        return self._columns

    @property
    def features(self) -> Mapping[str, float]:
        return {
            f'{col}_{lag:d}': self._buffer[lag][col]
            for lag, col in itertools.product(range(self.n_lags), self.columns)
        }

    def insert(self, row: Mapping[str, float]) -> None:
        self._buffer.appendleft(row)

    def iterate(self, rows: Iterable[Mapping[str, float]],
                ) -> Generator[Mapping[str, float], None, None]:
        '''Yield rows with lagged observations based on rows.

        Parameters
        ----------
        rows: Iterator of rows.

        Returns
        -------
        Generator yielding rows with lagged observations. The initial state of
        the object will influence the first `n_lags` observations, and the
        internal state of the object will be modified.
        '''
        for row in rows:
            self.insert(row)
            yield self.features


def test_lag_features():
    data = pd.DataFrame({
        'A': list(range(10)),
        'B': list(range(10, 20)),
    })

    lf = LagFeatures(2, data.columns)
    assert set(lf.feature_columns) == {'A_0', 'A_1', 'B_0', 'B_1'}

    # No data inserted yet
    assert pd.Series(lf.features).isna().all()

    # Some data inserted, but not enough to remove all NaNs
    lf.insert(data.iloc[0])
    nans = pd.Series(lf.features).isna()
    print(lf.features)
    assert nans.any()
    assert not nans.all()
    assert nans[['A_1', 'B_1']].all()
    assert not nans[['A_0', 'B_0']].any()
    assert lf.features['A_0'] == 0
    assert lf.features['B_0'] == 10

    # Enough data inserted to fill the buffer
    lf.insert(data.iloc[1])
    assert not pd.Series(lf.features).isna().any()
    assert lf.features == {
        'A_0': 1,
        'B_0': 11,
        'A_1': 0,
        'B_1': 10,
    }


def test_lag_features_iterate():
    data = pd.DataFrame({
        'A': list(range(0, 10)),
        'B': list(range(10, 20)),
    })
    # A single lagged feature means an identify transformation
    lf = LagFeatures(1, data.columns)
    lagged_data = (pd.DataFrame(
        list(lf.iterate(row for _, row in data.iterrows())))
        .rename(columns={'A_0': 'A', 'B_0': 'B'})
    )
    assert (data == lagged_data).all().all()


class RCForest:

    def __init__(
            self, n_trees=100, tree_size=256,
            scaler_factory=preprocessing.RobustScaler):
        self._trees = [rrcf.RCTree() for _ in range(n_trees)]
        self._tree_size = tree_size
        self._scaler = scaler_factory()
        self._point_index = 0

    @property
    def tree_size(self):
        return self._tree_size

    @property
    def shingle_size(self):
        return self._shingle_size

    @property
    def n_trees(self):
        return len(self._trees)

    def initialize(self, data):
        self._scaler.fit(data)
        for point in data:
            _ = self.insert_point(point)


    def insert_point(self, point):
        '''Insert a point into the tree and report its anomaly score.'''
        codisp_sum = 0
        point_scaled = self._scaler.transform(point.reshape((1, -1)))
        for tree in self._trees:
            if len(tree.leaves) > self.tree_size:
                tree.forget_point(self._point_index - self.tree_size)
            tree.insert_point(point_scaled, index=self._point_index)
            codisp_sum += tree.codisp(self._point_index)

        self._point_index += 1
        codisp = codisp_sum / self.n_trees
        return codisp
