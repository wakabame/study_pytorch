from abc import ABC, abstractmethod
from typing import Any, Self

import pandas as pd
from nptyping import Float, NDArray

class Classifier(ABC):

    @abstractmethod
    def fit(self, X: NDArray[Any, Float], y: pd.Series) -> Self:
        ...

    @abstractmethod
    def predict(self, X: NDArray[Any, Float]) -> NDArray[Any, Float]:
        ...
