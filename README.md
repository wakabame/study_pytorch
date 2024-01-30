# study_pytorch

## 目的

このリポジトリでは、教科書「Python 機械学習プログラミング [PyTorch & scikit-learn 編] の勉強をしていくための個人的なメモである。

ついでに、 Python のコード管理や CI/CD の勉強も兼ねている

## 教科書との違い

教科書では `pip` や `conda` によるパッケージ管理を進めているが、勉強のために `rye` を用いる

## ライセンス

自分の学習のためのコードなので未定義

## 学習の仕方

``` sh
source .venv/bin/activate
jupyter notebook
```

## CelabA データのダウンロード

```0B7EVK8r0v71pZjFTYXZWM3FlRnM
FILE_ID=0B7EVK8r0v71pZjFTYXZWM3FlRnM
FILE_NAME=img_align_celeba.zip
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILE_ID}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${FILE_ID}" -o celeba/${FILE_NAME}
```
