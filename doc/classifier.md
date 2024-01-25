# 分類器の数学的背景

## モチベーション

二値関数 $f: \mathbb{R}^m \rightarrow \lbrace 0, 1 \rbrace$ を未知とする.
$N$ 個の訓練データ $(x_n)_{n=1}^N \subset \mathbb{R}^m$, $(y_n)_{n=1}^N \subset \lbrace 0, 1 \rbrace$ を用いて近似する手段を考えたい.

特に, 重み $w \in \mathbb{R}^m$, $b \in \mathbb{R}$ を用いた, 以下のような関数(分類器と呼ぶ) を構成することを考える;

$$
F = F_{w,b}:\mathbb{R}^m \rightarrow \lbrace 0, 1 \rbrace.
$$

重みは訓練データとアルゴリズムにしたがって決定されるものであり, 分類器の性質やアルゴリズムによって分類器の名称が異なっている. 分類器はしばしば背後に数理モデルと良し悪しを測る指標が背後にあり, 実際はそれらによってアルゴリズムが構成されている.

ここからはいくつかの分類器とそのアルゴリズムについて解説する.

## パーセプトロン

関数 $\sigma: \mathbb{R} \rightarrow \lbrace 0, 1 \rbrace$ を
$$
\begin{align*}
\sigma(z) = \begin{cases} 1, & \text{if} & z\ge 0 \cr
0, & \text{if} & z < 0
\end{cases}
\end{align*}
$$
により定める((これをパーセプトロンの決定関数と呼ぶ)).

### パーセプトロンの分類器

重み $w \in \mathbb{R}^m$, $b \in \mathbb{R}$ に応じて, 以下により定める.

$$
F(x) = \sigma(\langle x, w\rangle +b)
$$

### パーセプトロンの重み決定のアルゴリズム

定数 $\eta \in (0,1)$ を固定する((これを学習率と呼ぶ)). 
重みの更新列 $(w_n)_{n=0}^N \subset \mathbb{R}^m$, $(b_n)_{n=0}^N \subset \mathbb{R}$ 及び予測値の列 $(\hat{y}_n)_{n=1}^N \subset \lbrace 0, 1 \rbrace$ を帰納的に次のように定める.

1. $w_0$, $b_0$ は $0$ または $0$ に近い乱数((なぜ??))で初期化する
2. 次の規則で逐次値を定める;

    任意の $t=1,2,3,\cdots$ について,
    $$
    \begin{align*}
    \hat{y}_t &= \sigma(\langle x, w_{t-1}\rangle + b_{t-1}),\cr
    w_{t}^j &= w_{t-1}^j +\eta (y_t - \hat{y}_t)x_t^j, \quad\text{for}\quad  j=1,2,\cdots ,m,\cr
    b_{t} &= b_{t-1} + \eta(y_t-\hat{y}_t).
    \end{align*}
    $$
3. 十分大きな $t=T$ で打ち切り, $w=w_T$, $b=b_T$ と置く

### パーセプトロンの特徴

* 訓練データが線形分離可能な場合, 繰り返すことによって $y_n = \hat{y}_n$ が任意の $n=1,2,\cdots, N$ に対して成り立つようになり, 重みの更新が行われなくなる.
* 訓練データが線形分離可能でない場合は, どんな重みに対してもある訓練データの元によって重みの更新がされるため, 重みが収束しない.

## ADALINE

### ADALINEの分類器

重み $w \in \mathbb{R}^m$, $b \in \mathbb{R}$ に応じて, 以下により定める.

$$
F(x) = \langle x, w\rangle +b.
$$

### ADALINEの重み決定のアルゴリズム

定数 $\eta \in (0,1)$ を固定する((これを学習率と呼ぶ)).
重みの更新列 $(w_n)_{n=0}^N \subset \mathbb{R}^m$, $(b_n)_{n=0}^N \subset \mathbb{R}$ を帰納的に次のように定める.

1. $w_0$, $b_0$ は $0$ または $0$ に近い乱数((なぜ??))で初期化する
2. 次の規則で逐次値を定める;

    任意の $t=1,2,3,\cdots$ について,
    $$
    \begin{align*}
    z_{n, t-1} &= \langle x_n, w_{t-1}\rangle +b_{t-1}, \cr
    w_{t}^j &= w_{t-1}^j +\eta \dfrac{1}{N}\sum_{n=1}^N(y_n - z_{n, t-1})x_n^j, \quad\text{for}\quad  j=1,2,\cdots ,m,\cr
    b_{t} &= b_{t-1} +\eta \dfrac{1}{N}\sum_{n=1}^N(y_n - z_{n, t-1}).
    \end{align*}
    $$
3. 十分大きな $t=T$ で打ち切り, $w=w_T$, $b=b_T$ と置く

## ADALINEの特徴

このアルゴリズムは分類器による二乗誤差により損失関数 $L$ を定めたときの勾配降下法となっている.
すなわち, $w \in \mathbb{R}^m$, $b \in \mathbb{R}$ に対して損失関数を

$$
L(w,b) := \dfrac{1}{2N} \sum_{n=1}^N \left( y_n -(\langle x_n, w\rangle +b)\right)^2
$$

としたときの勾配降下法となっている. 実際, 勾配降下法によると,

$$
\begin{align*}
w_{t}^j &= w_{t-1}^j - \eta \dfrac{\partial L}{\partial w^j}({w_{t-1},b_{t-1}}), \quad\text{for}\quad  j=1,2,\cdots ,m,\cr
b_{t} &= b_{t-1} - \eta \dfrac{\partial L}{\partial b}({w_{t-1},b_{t-1}}).
\end{align*}
$$

による更新が求められ, 微分計算を行うことによりアルゴリズムと一致している.