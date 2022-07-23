import matplotlib.pyplot as plt
import pandas as pd

# CSVを読み込む
df = pd.read_csv("kion10y.csv", encoding="utf-8")

# 月ごとに平均を求める
gb = df.groupby(['月'])["気温"]
gg = gb.sum() / gb.count()

print(gg)
gg.plot()
plt.savefig("tenki-heikin-tuki.png")
plt.show()