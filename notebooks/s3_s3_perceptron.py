# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import matplotlib.pyplot as plt

# 学習データを読み込む

train = np.loadtxt('s3_s3_images.csv', delimiter=',', skiprows=1)
train_x = train[:,0:2]
train_y = train[:,2]

# plot
plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
plt.axis('scaled')
plt.show()

# +
# 重みの初期化
w = np.random.rand(2)

# 識別関数
def f(x):
    if np.dot(w, x) >= 0:
        return 1
    else:
        return -1

# 繰り返し回数
epoch = 10

# 更新回数
count = 0

# 重みを学習する
for _ in range(epoch):
    for x, y in zip(train_x, train_y):
        if f(x) != y:
            # 重みベクトルを分類があってない場合に少しづつづらすことで決定境界を近づけていく
            w = w + y * x
            # ログの出力
            count += 1
            print('{}回目： w = {}'.format(count, w))

# +
            
x1 = np.arange(0, 500) 

plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
plt.plot(x1, -w[0] / w[1] * x1, linestyle='dashed')
plt.show()


# +
# 適当に分類させてみる
# 200x100の横長画像

f([200, 100])
# -

# 100x200の縦長画像
f([100, 200])


