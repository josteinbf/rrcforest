import pandas as pd

from rrcforest import LagFeatures


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
