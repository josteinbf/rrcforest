from typing import Sequence, Union, Iterable

from rrcf import rrcf
import numpy as np


class RCForest:
    _forest: Sequence[rrcf.RCTree]
    _tree_size: int
    _point_index: int

    def __init__(
            self, n_trees: int = 100, tree_size: int = 256,
            random_state: Union[None, int, np.random.RandomState] = None):
        if random_state is None:
            rng = np.random.RandomState()
        elif isinstance(random_state, int):
            rng = np.random.RandomState(random_state)
        else:
            rng = random_state
        max_seed = np.iinfo(np.int32).max
        self._forest = [
            rrcf.RCTree(random_state=rng.randint(max_seed))
            for _ in range(n_trees)
        ]
        self._tree_size = tree_size
        self._point_index = 0

    @property
    def tree_size(self) -> int:
        return self._tree_size

    @property
    def n_trees(self) -> int:
        return len(self._forest)

    def insert_batch(self, points: Iterable[np.ndarray]):
        return np.asarray(
            [self.insert(point) for point in points])

    def insert(self, point: np.ndarray):
        '''Insert a point into the tree and report its anomaly score.'''
        if point.ndim != 1:
            raise ValueError('expected 1D array')

        codisp_sum = 0
        for tree in self._forest:
            if len(tree.leaves) > self.tree_size:
                tree.forget_point(self._point_index - self.tree_size)
            tree.insert_point(point, index=self._point_index)
            codisp_sum += tree.codisp(self._point_index)

        self._point_index += 1
        codisp = codisp_sum / self.n_trees
        return codisp
