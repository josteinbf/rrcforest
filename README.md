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

## Development

To work on the code, get set up with

```sh
python -m pip install -e ".[test]"
```

To run tests and static checks, run

```sh
mypy -m rrcforest    # Static check of type hints
flake8               # Code formatting etc
pytest tests         # Tests
```

Contributions are welcome in the form of pull requests!
