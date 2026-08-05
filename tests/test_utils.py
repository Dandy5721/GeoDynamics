import numpy as np

from GeoDynamics.utils import sliding_window_corrcoef


def test_sliding_window_corrcoef_shape_and_finiteness():
    sequence = np.arange(30, dtype=np.float32).reshape(3, 10)
    matrices = sliding_window_corrcoef(sequence, window_size=5)
    assert matrices.shape == (10, 3, 3)
    assert np.isfinite(matrices).all()
    np.testing.assert_allclose(matrices, matrices.transpose(0, 2, 1), atol=1e-6)
