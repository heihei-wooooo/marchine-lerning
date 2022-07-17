from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# データを読み込む
wine = pd.read_csv("winequality-white.csv", sep=";", encoding="utf-8")
# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/ からCSVファイル取得

# データをラベルとデータに分離
y = wine["quality"]
x = wine.drop("quality", axis=1)

# yラベル
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <=7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

# 学習用とテスト用に分割する
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 学習する
model = RandomForestClassifier()
model.fit(x_train, y_train)

# 評価する
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("正解率 = ", accuracy_score(y_test, y_pred))

# 品質データごとにグループ分けして、その数を数える
count_data = wine.groupby("quality")["quality"].count()
print(count_data)

# 数えたデータをグラフに描画
count_data.plot()
plt.savefig("wine-count-plt.png")
plt.show()