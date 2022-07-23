import matplotlib.pyplot as plt
import pandas as pd

# ファイルを読み込む
df = pd.read_csv("kion10y.csv", encoding="utf-8")
# 気温が30度超えのデータを調べる
over30_bool = (df["気温"] > 30)
# データを抽出
over30 = df[over30_bool]
# 年ごとにカウント
cnt = over30.groupby(["年"])["年"].count()

print(cnt)
cnt.plot()
plt.savefig("tenki-over30.png")
plt.show()