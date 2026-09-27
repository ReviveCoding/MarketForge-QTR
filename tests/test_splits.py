import numpy as np

from marketforge.splits import _safe_stats


def test_scaler_handles_constant_and_nan_columns() -> None:
    matrix = np.array([[1.0, np.nan], [1.0, np.nan]])
    mean, scale = _safe_stats(matrix)
    assert np.isfinite(mean).all()
    assert np.isfinite(scale).all()
    assert (scale > 0).all()
