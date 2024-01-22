from typing import Any, Self

import numpy as np
import pandas as pd
from nptyping import Float, NDArray


class Perceptron:
    def __init__(
        self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1
    ) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: NDArray[Any, Float], y: pd.Series) -> Self:
        rgen = np.random.default_rng(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y, strict=True):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self, X: NDArray[Any, Float]) -> float:
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: NDArray[Any, Float]) -> NDArray[Any, Float]:
        return np.where(self.net_input(X) >= 0.0, 1, 0)
