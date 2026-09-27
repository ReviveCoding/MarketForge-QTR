import numpy as np

from marketforge.baselines import _metrics


def test_metrics_perfect_prediction() -> None:
    target = np.array([-1, 0, 1])
    metrics = _metrics(target, target.copy())
    assert metrics == {"macro_f1": 1.0, "mcc": 1.0, "balanced_accuracy": 1.0}
