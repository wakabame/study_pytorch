from abc import ABC, abstractmethod
from typing import Any
from typing import TypeVar

from nptyping import Float, NDArray

Self = TypeVar("Self", bound="Classifier")


class Classifier(ABC):
    @abstractmethod
    def fit(self: Self, X: NDArray[Any, Float], y: NDArray[Any, Float]) -> Self:
        ...

    @abstractmethod
    def predict(self: Self, X: NDArray[Any, Float]) -> NDArray[Any, Float]:
        ...
