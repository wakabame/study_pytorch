from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from nptyping import Bool, Float, NDArray

from study_pytorch.classifier import Classifier


def plot_decision_regions(
    X: NDArray[Any, Float],
    y: NDArray[Any, Float],
    classifier: Classifier,
    test_idx: NDArray[Any, Bool] | None = None,
    resolution: float = 0.02,
    x_label: str | None = None,
    y_label: str | None = None,
) -> None:
    makers = ("o", "s", "^", "v", "<")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    # 各特徴量を1次元配列に変換して予測を実行
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 予測結果を元のグリッドポイントのデータサイズに変換
    lab = lab.reshape(xx1.shape)
    # グリッドポイントの等高線をプロット
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)  # 軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとに訓練データをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=makers[idx],
            label=f"Class: {cl}",
            edgecolor="black",
        )

    # テストデータ点を目立たせる（点を〇で表示）
    if test_idx:
        # 全てのデータ点をプロット
        X_test, _ = X[test_idx, :], y[test_idx]
        plt.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c="none",
            edgecolor="black",
            alpha=1.0,
            linewidths=1,
            marker="o",
            s=100,
            label="Test set",
        )

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def easy_plot(plot_target: Callable) -> None:
    """グラフを描画します"""
    z = np.arange(-7, 7, 0.1)
    sigma_z = plot_target(z)

    plt.plot(z, sigma_z)
    plt.axvline(0.0, color="k")  # 水r直線を追加
    plt.xlabel("x")
    plt.ylabel("f(x)")

    ax = plt.gca()
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.show()
