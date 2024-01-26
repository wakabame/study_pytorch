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
\begin{aligned}
\sigma(z) = \begin{cases} 1, & \text{if} & z\ge 0 \cr
0, & \text{if} & z < 0
\end{cases}
\end{aligned}
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
    \begin{aligned}
    \hat{y}_t &= \sigma(\langle x, w_{t-1}\rangle + b_{t-1}),\cr
    w_{t}^j &= w_{t-1}^j +\eta (y_t - \hat{y}_t)x_t^j, \quad\text{for}\quad  j=1,2,\cdots ,m,\cr
    b_{t} &= b_{t-1} + \eta(y_t-\hat{y}_t).
    \end{aligned}
    $$
3. 十分大きな $t=T$ で打ち切り, $w=w_T$, $b=b_T$ と置く

### パーセプトロンの特徴

* 訓練データが線形分離可能な場合, 繰り返すことによって $y_n = \hat{y}_n$ が任意の $n=1,2,\cdots, N$ に対して成り立つようになり, 重みの更新が行われなくなる.
* 訓練データが線形分離可能でない場合は, どんな重みに対してもある訓練データの元によって重みの更新がされるため, 重みが収束しない.

## ADALINE

### ADALINEの分類器

重み $w \in \mathbb{R}^m$, $b \in \mathbb{R}$ に応じて, 以下により定める.

$$
\begin{aligned}
F(x) = \begin{cases}
    1\quad & \text{if}\quad \langle x, w\rangle +b \ge \frac{1}{2},\cr
    0\quad & \text{if}\quad \langle x, w\rangle +b < \frac{1}{2}.
    \end{cases}
\end{aligned}
$$

### ADALINEの重み決定のアルゴリズム

定数 $\eta \in (0,1)$ を固定する((これを学習率と呼ぶ)).
重みの更新列 $(w_n)_{n=0}^N \subset \mathbb{R}^m$, $(b_n)_{n=0}^N \subset \mathbb{R}$ を帰納的に次のように定める.

1. $w_0$, $b_0$ は $0$ または $0$ に近い乱数((なぜ??))で初期化する
2. 次の規則で逐次値を定める;

    任意の $t=1,2,3,\cdots$ について,
    $$
    \begin{aligned}
    z_{n, t-1} &= \langle x_n, w_{t-1}\rangle +b_{t-1}, \cr
    w_{t}^j &= w_{t-1}^j +\eta \dfrac{1}{N}\sum_{n=1}^N(y_n - z_{n, t-1})x_n^j, \quad\text{for}\quad  j=1,2,\cdots ,m,\cr
    b_{t} &= b_{t-1} +\eta \dfrac{1}{N}\sum_{n=1}^N(y_n - z_{n, t-1}).
    \end{aligned}
    $$
3. 十分大きな $t=T$ で打ち切り, $w=w_T$, $b=b_T$ と置く

## ADALINEの特徴

このアルゴリズムは分類器による二乗誤差により損失関数 $L$ を定めたときの勾配降下法となっている.
すなわち, $w \in \mathbb{R}^m$, $b \in \mathbb{R}$ に対して損失関数を

$$
L(w,b) := \dfrac{1}{2N} \sum_{n=1}^N \left( y_n -(\langle x_n, w\rangle +b)\right)^2
$$

としたときの勾配降下法となっている. 実際, 勾配降下法によると, 各計算ステップ $t=1,2,\cdots$ に対して

$$
\begin{aligned}
w_{t}^j &= w_{t-1}^j - \eta \dfrac{\partial L}{\partial w^j}({w_{t-1},b_{t-1}}), \quad\text{for}\quad  j=1,2,\cdots ,m,\cr
b_{t} &= b_{t-1} - \eta \dfrac{\partial L}{\partial b}({w_{t-1},b_{t-1}}).
\end{aligned}
$$

による更新が求められ, 微分計算を行うことによりアルゴリズムと一致している.

## ロジスティック回帰

「回帰」と名前がついているが分類に使われる.
また, 関数 $\sigma: \mathbb{R} \rightarrow (0, 1)$ を
$$
\sigma(z) = \dfrac{1}{1+\exp{(-z)}}
$$
により定める((これをシグモイド関数と呼ぶ)).

### ロジスティック回帰の分類器

重み $w \in \mathbb{R}^m$, $b \in \mathbb{R}$ に応じて, 以下により定める.

$$
\begin{aligned}
F(x) = \begin{cases}
    1\quad & \text{if}\quad \sigma(\langle x, w\rangle +b) \ge \frac{1}{2},\cr
    0\quad & \text{if}\quad \sigma(\langle x, w\rangle +b) < \frac{1}{2}
    \end{cases}
\end{aligned}
$$

### ロジスティック回帰の重み決定のアルゴリズム

定数 $\eta \in (0,1)$ を固定する((これを学習率と呼ぶ)).
重みの更新列 $(w_n)_{n=0}^N \subset \mathbb{R}^m$, $(b_n)_{n=0}^N \subset \mathbb{R}$ を帰納的に次のように定める.

1. $w_0$, $b_0$ は $0$ または $0$ に近い乱数((なぜ??))で初期化する
2. 次の規則で逐次値を定める;

    任意の $t=1,2,3,\cdots$ について,
    $$
    \begin{aligned}
    z_{n, t-1} &= \langle x_n, w_{t-1}\rangle +b_{t-1}, \cr
    w_{t}^j &= w_{t-1}^j +\eta \dfrac{1}{N}\sum_{n=1}^N(y_n - \sigma(z_{n, t-1}))x_n^j, \quad\text{for}\quad  j=1,2,\cdots ,m,\cr
    b_{t} &= b_{t-1} +\eta \dfrac{1}{N}\sum_{n=1}^N(y_n - \sigma(z_{n, t-1})).
    \end{aligned}
    $$
3. 十分大きな $t=T$ で打ち切り, $w=w_T$, $b=b_T$ と置く

### ロジスティック回帰の特徴

ADALINE とはシグモイド関数の代わりに, $\sigma(z) = z$ (恒等写像)とすることでアルゴリズムは一致する. そのため, 実装上は「活性化関数」のみが差分となる. ADALINE と比較すると, 次に述べるような確率モデルに対する尤度最大化のアルゴリズムとなっている.

#### 確率モデル

$X, Y$ をそれぞれ $\mathbb{R}^m$ $\lbrace 0, 1\rbrace$ に値をとる確率変数とする. ここで, $p = p_x= P(Y=1|X=x)$ とし, パラメータ $w \in \mathbb{R}^m$, $b \in \mathbb{R}$ を持つような以下のような確率分布を持つものとする;

$$
\log\dfrac{p}{1-p} = \langle x, w \rangle +b.
$$

先に述べたシグモイド関数を使って書き直すと, 

$$
p = \sigma(\langle x, w \rangle +b)
$$

である. 

仮に, この確率分布に従う独立なサンプルとして $(x_n)_{n=1}^N \subset \mathbb{R}^m$, $(y_n)_{n=1}^N \subset \lbrace 0, 1 \rbrace$ が得られたとすると, このときの尤度関数 $\mathcal{L}$ は, $z_n=\langle x, w \rangle +b$ を用いて次のように書き表せる

$$
\begin{aligned}
\mathcal{L}\left(w,b \mid (x_n)_{n=1}^N, (y_n)_{n=1}^N\right)
   &= P((Y=y_n)_{n=1}^N \mid (X=x_n)_{n=1}^N ; w,b)\cr
   &= \prod_{n=1}^N P(Y=y_n \mid X=x_n w,b)\cr
   &= \prod_{n=1}^N \sigma(z_n)^{y_n} \left(1-\sigma(x_n)\right)^{1-y_n}.
\end{aligned}
$$

したがって, 対数尤度関数 $\mathcal{l}$ は

$$
\mathcal{l}\left(w,b \mid (x_n)_{n=1}^N, (y_n)_{n=1}^N\right)
= \sum_{n=1}^N\left( y_n \log\sigma(z_n) + (1-y_n)\log\sigma(1-z_n)\right)
$$

であり, 損失関数 $L(w, b)$ はそれに $-1/N$ を掛けて

$$
L(w,b)
= \dfrac{1}{N}\sum_{n=1}^N \Big( -y_n \log\sigma(z_n) - (1-y_n)\log\sigma(1-z_n)\Big)
$$

と定めることにより, 損失関数に対する最小化問題は尤度関数の最大化問題と一致していることがわかる.

さらに, この損失関数に対する勾配降下法がアルゴリズムと一致していることを確かめよう. $\sigma'(z) = \sigma(z)(1-\sigma(z))$ に注意すると, 合成関数の微分則を適用して

$$
\begin{aligned}
&\dfrac{\partial}{\partial w^j} \Big( -y_n\log \sigma(z_n)\Big)
    = -\dfrac{y_n}{\sigma(z_n)}\sigma(z_n)(1-\sigma(z_n)) x^j,\cr
&\dfrac{\partial}{\partial w^j} \Big( -(1-y_n)\log (1-\sigma(z_n))\Big)
    = \dfrac{1-y_n}{1-\sigma(z_n)}\sigma(z_n)(1-\sigma(z_n)) x^j
\end{aligned}
$$

を得る. まとめると

$$
\dfrac{\partial L}{\partial w^j}(w,b) = \dfrac{1}{N}(-y_n +\sigma(z_n))x^j
$$

となり, アルゴリズムで述べた $w^j$ 方向の勾配降下法での更新式が得られる. また, $b$ 方向の微分も同様に計算される.
