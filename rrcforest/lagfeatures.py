from typing import (
    Sequence, Mapping, Deque, Iterable, Generator, Optional, TypeVar, Generic)
import itertools
from collections import deque


T = TypeVar('T')


class LagFeatures(Generic[T]):
    _buffer: Deque[Mapping[str, Optional[T]]]
    _columns: Sequence[str]

    def __init__(self, n_lags: int, columns: Sequence[str]):
        self._columns = columns
        self._buffer = deque(
            n_lags * [{col: None for col in columns}],
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
    def features(self) -> Mapping[str, Optional[T]]:
        return {
            f'{col}_{lag:d}': self._buffer[lag][col]
            for lag, col in itertools.product(range(self.n_lags), self.columns)
        }

    def insert(self, row: Mapping[str, Optional[T]]) -> None:
        self._buffer.appendleft(row)

    def iterate(self, rows: Iterable[Mapping[str, Optional[T]]],
                ) -> Generator[Mapping[str, Optional[T]], None, None]:
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
