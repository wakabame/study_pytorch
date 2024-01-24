from abc import ABC, abstractmethod
from typing import Any, Self

from nptyping import Float, NDArray


class Classifier(ABC):
    @abstractmethod
    def fit(self, X: NDArray[Any, Float], y: NDArray[Any, Float]) -> Self:
        ...

    @abstractmethod
    def predict(self, X: NDArray[Any, Float]) -> NDArray[Any, Float]:
        ...
