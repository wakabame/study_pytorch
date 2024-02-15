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

待てば大体大丈夫。ダウンロードがどうしてもうまくいかないとき

``` sh
FILE_ID=0B7EVK8r0v71pZjFTYXZWM3FlRnM
FILE_NAME=img_align_celeba.zip
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILE_ID}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${FILE_ID}" -o celeba/${FILE_NAME}
cd celeba
unzip $FILE_NAME
cd -
```

## collaboratory でライブラリとして実行する

GPU を無料で使える colab 環境を利用することができる
GitHub から clone して, `requirements-colab.txt` により依存を入手して使用することができる

```
!git clone https://github.com/wakabame/study_pytorch.git
# 処理とは関係ないファイルはすべて削除する
!mv ./study_pytorch ./study_pytorch_
!mv ./study_pytorch_/study_pytorch .
!mv ./study_pytorch_/requirements-colab.txt .
!rm -rf ./study_pytorch_
!pip install -r requirements-colab.txt
```

現在 colab 環境のバージョンは Python=3.10.12 なのでそれに合わせてこのリポジトリでも Python=3.10 を前提としている

以下のコマンドで確認できる
```
!python --version  # Python 3.10.12
!python -c 'import torch; print(torch.__version__)'  # 2.1.0+cu121
!python -c 'import torchvision; print(torchvision.__version__)'  # 0.16.0+cu121
```
