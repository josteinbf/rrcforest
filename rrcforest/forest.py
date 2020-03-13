from typing import Sequence, Mapping, Deque, Iterable, Generator
import itertools
from collections import deque

from rrcf import rrcf
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


class RCForest:

    def __init__(self, n_trees=100, tree_size=256):
        self._trees = [rrcf.RCTree() for _ in range(n_trees)]
        self._tree_size = tree_size
        self._point_index = 0

    @property
    def tree_size(self):
        return self._tree_size

    @property
    def n_trees(self):
        return len(self._trees)

    def initialize(self, data):
        for point in data:
            _ = self.insert_point(point)

    def insert_point(self, point):
        '''Insert a point into the tree and report its anomaly score.'''
        codisp_sum = 0
        for tree in self._trees:
            if len(tree.leaves) > self.tree_size:
                tree.forget_point(self._point_index - self.tree_size)
            tree.insert_point(point, index=self._point_index)
            codisp_sum += tree.codisp(self._point_index)

        self._point_index += 1
        codisp = codisp_sum / self.n_trees
        return codisp
