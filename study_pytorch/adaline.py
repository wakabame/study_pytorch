from typing import Any, Self

import numpy as np
from nptyping import Float, NDArray


from study_pytorch.classifier import Classifier


class AdalineGD(Classifier):
    """ADAptive LInear NEuron 分類器

    Parameters
    ----------
    eta : float
        学習率 (0.0 より大きく 1.0 以下の値)
    n_iter : int
        訓練データの訓練回数
    random_state : int
        重みを初期化するための乱数シード

    Attribute
    ----------
    w_ : 1次元配列
        適合後の重み
    b_ : スカラー
        適合後のバイアス
    losses_ : リスト
        各エポックでの MSE 誤差関数の値
    """
    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: NDArray[Any, Float], y: NDArray[Any, Float]) -> Self:
        """訓練データに適合させる

        Parameters
        ----------
        X : NDArray[Any, Float]
        y : NDArray[Any, Float]

        Returns
        -------
        Self
        """
        rgen = np.random.default_rng(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output

            self.w_ += self.eta * X.T.dot(errors)/X.shape[0]
            self.b_ += self.eta * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X: NDArray[Any, Float]) -> NDArray[Any, Float]:
        return np.dot(X, self.w_) + self.b_

    def activation(self, X: NDArray[Any, Float]) -> NDArray[Any, Float]:
        return X

    def predict(self, X: NDArray[Any, Float]) -> NDArray[Any, Float]:
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
