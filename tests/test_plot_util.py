from typing import Any

import numpy as np
from nptyping import Float, NDArray

from study_pytorch.plot_util import easy_plot


def sigmoid(x: NDArray[Any, Float]) -> NDArray[Any, Float]:
    return 1.0 / (1.0 + np.exp(-x))


def test_easy_plot() -> None:
    easy_plot(sigmoid)
