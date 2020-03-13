import numpy as np

from rrcforest import RCForest


def test_constants():
    n_observations = 300
    n_features = 4
    data = np.ones(shape=(n_observations, n_features), dtype=float)
    data *= 3.1415

    forest = RCForest(random_state=42)
    scores = forest.insert_batch(data)

    assert scores.shape == (n_observations,)
    assert np.all(scores == 0.)


def test_random():
    n_observations = 60
    n_features = 5
    data = np.random.uniform(size=(n_observations, n_features))
    print(data.shape)
    forest = RCForest(random_state=42)
    scores = forest.insert_batch(data)

    assert scores.shape == (n_observations,)
    assert scores[0] == 0.
    assert np.all(scores[1:] != 0.)
