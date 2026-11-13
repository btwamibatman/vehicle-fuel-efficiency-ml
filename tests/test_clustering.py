import numpy as np
import pandas as pd
from vehicle_efficiency.clustering import FEATURES, build_clusterer, search


def test_fit_uses_only_engine_features_and_imputes_missing():
    frame = pd.DataFrame({'cylinders':[4.,4.,6.,6.,8.,8.], 'displacement':[1.5,np.nan,3.,3.5,5.,5.5], 'mpg':[99,1,99,1,99,1]})
    pipe = build_clusterer(3).fit(frame[FEATURES])
    a = pipe.predict(frame[FEATURES]); frame['mpg'] *= -100
    b = build_clusterer(3).fit_predict(frame[FEATURES])
    assert np.array_equal(a,b)
    assert pipe['impute'].statistics_[1] == 3.5


def test_search_ignores_target():
    rng = np.random.RandomState(42)
    frame = pd.DataFrame({'cylinders':rng.choice([4,6,8],100), 'displacement':rng.uniform(1,6,100), 'mpg':rng.uniform(10,50,100)})
    first=search(frame); frame['mpg'] = frame.mpg.iloc[::-1].to_numpy()
    pd.testing.assert_frame_equal(first,search(frame))
