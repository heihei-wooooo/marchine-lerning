from mozaiku import mozaiku as mo
import matplotlib.pyplot as plt
import cv2

# カスケードファイルを指定して分類器を作成
cascade_file = "opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)

# 画像を読み込んでグレースケールに変換
img = cv2.imread("img.jpeg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 顔検出を実行
face_list = cascade.detectMultiScale(img_gray, minSize=(150, 150))
if len(face_list) == 0: quit()

# 認識した部分の画像にモザイクをかける
for (x, y, w, h) in face_list:
    img = mo(img, (x, y, x+w, y+h), 10)

# モザイクをかけた画像を出力
cv2.imwrite("mozaiku-face.png", img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()