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

import numpy as np
import matplotlib.pyplot as plt
x = np.random.randn(1000)
y = np.random.randn(1000)
plt.scatter(x, y)
plt.show()

plt.hist(x)

plt.hist(y)
