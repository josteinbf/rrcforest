from typing import Sequence, Union

from rrcf import rrcf
import numpy as np


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
