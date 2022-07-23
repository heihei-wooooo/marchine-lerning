from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ファイルの読み込み
df = pd.read_csv("kion10y.csv", encoding="utf-8")

# データを学習用とテスト用に分割する
train_year = (df["年"] <= 2016)
test_year = (df["年"] >= 2017)
interval = 6

# 過去6日分を学習するデータを作成
def make_data(data):
    x = [] # 学習データ
    y = [] # 結果
    temps = list(data["気温"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return(x,y)

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])

# 線形回帰分析を行う
lr = LinearRegression(normalize=True)
lr.fit(train_x, train_y) # 学習
pre_y = lr.predict(test_x) # 予測
print(1)

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(test_y, c='r')
plt.plot(pre_y, c='b')
plt.savefig('tenki-kion-lr.png')
print(2)
plt.show()
print(3)

diff_y = abs(pre_y - test_y)
print(4)

print("average=", sum(diff_y) / len(diff_y))
print(5)

print("max=", max(diff_y))
print(6)

