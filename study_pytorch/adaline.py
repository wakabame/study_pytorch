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

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int | None = 1) -> None:
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

            self._update_weights(X, y, output)
            self.losses_.append(self._loss(X, y, output))

        return self

    def _update_weights(self, X: NDArray[Any, Float], y: NDArray[Any, Float], output: NDArray[Any, Float]) -> None:  # type: ignore[no-untyped-def]
        """ADALINE の学習規則を使って重みを更新"""
        errors = y - output
        self.w_ += self.eta * X.T.dot(errors) / X.shape[0]
        self.b_ += self.eta * errors.mean()


    def _loss(self, X: NDArray[Any, Float], y: NDArray[Any, Float], output: NDArray[Any, Float]) -> float:
        """二乗誤差平均 MSE で算出する
        """
        errors = y - output
        return (errors**2).mean()


    def net_input(self, X: NDArray[Any, Float]) -> NDArray[Any, Float]:
        return np.dot(X, self.w_) + self.b_

    def activation(self, X: NDArray[Any, Float]) -> NDArray[Any, Float]:
        return X

    def predict(self, X: NDArray[Any, Float]) -> NDArray[Any, Float]:
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


class AdalineSGD(AdalineGD):
    """ADAptive LInear NEuron 分類器

    Parameters
    ----------
    eta : float
        学習率 (0.0 より大きく 1.0 以下の値)
    n_iter : int
        訓練データの訓練回数
    shuffle : bool (defaults: True)
        True の場合は循環を回避するためにエポックごとに訓練データをシャッフル
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

    def __init__(
        self, eta: float = 0.01, n_iter: int = 10, shuffle: bool = True, random_state: int | None = None
    ) -> None:
        super().__init__(eta, n_iter, random_state)
        self.w_initialized = False
        self.shuffle = shuffle
        self.rgen = np.random.default_rng(self.random_state)

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
        self._initialize_weights(X.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            # 各訓練データの損失値を格納するリストを作成
            losses: list[float] = []
            for xi, target in zip(X, y, strict=True):
                # 特徴量 xi と目的変数 y を使った重みの更新と損失値の計算
                losses.append(self._update_weights(xi, target))

            # 訓練データの平均損失値の計算
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)

        return self

    def partial_fit(self, X: NDArray[Any, Float], y: NDArray[Any, Float]) -> Self:
        """重みを再初期化することなく訓練データに適合させる"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        # 目的変数の要素数が2以上の場合は各訓練データの特徴量 xi と目的変数 target で重みを更新
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y, strict=True):
                self._update_weights(xi, target)
        # 目的変数の要素数が1の場合は訓練データの特徴量Xと目的変数yで重みを更新
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(
        self, X: NDArray[Any, Float], y: NDArray[Any, Float]
    ) -> tuple[NDArray[Any, Float], NDArray[Any, Float]]:
        """訓練データをシャッフル"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m: int) -> None:
        """重みを小さな乱数で初期化"""
        self.rgen = np.random.default_rng(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.0)
        self.w_initialized = True

    def _update_weights(self, xi, target) -> float:  # type: ignore[no-untyped-def]
        """ADALINE の学習規則を使って重みを更新"""
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * xi * error
        self.b_ += self.eta * error
        loss = error**2
        return loss
