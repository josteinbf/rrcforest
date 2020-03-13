# Robust Random Cut Forest

The Robust Random Cut Forest is an anomaly detection algorithm well suited for
streaming data. This repo just adds a thin layer on top of what exists in the
[rrcf package](https://github.com/kLabUM/rrcf). The contributions here are:

1. An `RCForest` class, which implements the Robust Random Cut Forest as
   described in the examples for the rrcf package. This uses the `RCTree`
   class from the rrcf package for all the heavy lifting.
2. A preprocessing class for engineering lagged features, which is often useful
   for time series.

## Installation

```sh
python -m pip install .
```

As always, a virtualenv of some sort is recommended.

## Using the anomaly detector

Here is an example of how to run the anomaly detector on randomly generated
data, storing the anomaly scores (higher score indicates higher chance of
being an anomaly):

```py
import numpy as np
from rrcforest import RCForest

n_observations = 60
n_features = 5
data = np.random.uniform(size=(n_observations, n_features))

forest = RCForest(random_state=42)
scores = np.asarray([forest.insert(obs) for obs in data])

assert scores[0] == 0.            # First observation always gets score of zero
assert np.all(scores[1:] != 0.)   # All other observations should be non-zero
assert scores.shape == (n_observations,)
```

Note that there is no separate training and inference/test stages for the RRCF
anomaly detector, instead, the algorithm uses _all_ points seen for training,
and an anomaly score is also calculated for all observations. Consequently, the
order in which the observations are fed to the anomaly detector influences the
results. Over time, the anomaly detector will forget the earliest seen
observations.

All features should be normalized/standardized before feeding them to the
anomaly detector.

## Development

To work on the code, get set up with

```sh
python -m pip install -e ".[test]"
```

Changes in the code directory are then immediately reflected in the package
without reinstallation.

To run tests and static checks, run

```sh
mypy -m rrcforest    # Static check of type hints
flake8               # Code formatting etc
pytest tests         # Tests
```

Contributions are welcome in the form of pull requests!
