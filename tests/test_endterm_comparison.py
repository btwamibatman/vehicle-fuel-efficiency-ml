import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from vehicle_efficiency.evaluation import compute_metrics


def test_group_cv_never_splits_duplicate_groups():
    groups=np.repeat(np.arange(12),2)
    for train,val in GroupKFold(4).split(np.zeros((24,1)), groups=groups):
        assert set(groups[train]).isdisjoint(groups[val])


def test_metrics_use_full_precision_predictions():
    m=compute_metrics(pd.Series([1.,2.]),np.array([1.04,2.04]))
    assert abs(m['mae']-.04)<1e-10
