from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 学習用のデータと結果の準備
#X Y
learn_data = [[0,0], [1,0], [0,1], [1,1]]
# X xor Y
learn_label = [0, 1, 1, 0]

# アルゴリズムの指定
clf = LinearSVC()

# 学習用データと結果の学習
clf.fit(learn_data, learn_label)

# テストデータによる予測
test_data = [[0,0], [1,0], [0,1], [1,1]]
test_label = clf.predict(test_data)

# テスト結果に評価
print(test_data, "の予測評価：", test_label)
print("正解率 = ", accuracy_score([0, 1, 1, 0], test_label))