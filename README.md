# Iris Data Setを Chainerで分類する

https://archive.ics.uci.edu/ml/datasets/iris

## 準備: データの加工

元のデータでは、分類後の情報は名前で提供されている。

    4.9,3.0,1.4,0.2,Iris-setosa
    6.4,3.2,4.5,1.5,Iris-versicolor
    7.1,3.0,5.9,2.1,Iris-virginica
    ...

これを数字に置き換えている

    4.9,3.0,1.4,0.2,0
    6.4,3.2,4.5,1.5,1
    7.1,3.0,5.9,2.1,2
    ...

## 学習

    python3 ./learn.py

## 検証

    python3 ./check.py
