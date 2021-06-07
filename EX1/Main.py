import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


fname = 'balance-scale.data'
# Attribute Information:
# 1. Class Name: 3 (L, B, R) -> CN
# 2. Left-Weight: 5 (1, 2, 3, 4, 5) -> LW
# 3. Left-Distance: 5 (1, 2, 3, 4, 5) -> LD
# 4. Right-Weight: 5 (1, 2, 3, 4, 5) -> RW
# 5. Right-Distance: 5 (1, 2, 3, 4, 5) -> RD
name_of_columns = ['CN', 'LW', 'LD', 'RW', 'RD']
df = pd.read_csv(fname, header=None, names=name_of_columns)
col_size = len(df['CN'])
feature1 = np.zeros((df.size, 1))
feature2 = np.zeros((df.size, 1))
# convert four attributes to two attributes
feature1 = (df['LW'] * df['LD']).values
feature2 = (df['RW'] * df['RD']).values
feature1 = feature1.reshape((col_size, 1))
feature2 = feature2.reshape((col_size, 1))
# a little bit movement because of overlapping
rand1 = np.random.uniform(low=0.7, high=1.3, size=(col_size, 1))
rand2 = np.random.uniform(low=0.7, high=1.3, size=(col_size, 1))
feature1 = feature1 * rand1
feature2 = feature2 * rand1
# plotting
color = df['CN'].map({'L': 0, 'B': 1, 'R': 2})
colormap = np.array(['c', 'r', 'orange'])
no_fig = 1
fig = plt.figure(no_fig)
no_fig += 1
ax = fig.add_subplot(1, 1, 1)
ax.scatter(feature1, feature2, c=colormap[color], s=30)
ax.set_title("Class L: Cyan | Class B: Red | Class R: Orange")
# ax.legend(['L', 'B', 'R'])
fig = plt.figure(no_fig)
df.boxplot(name_of_columns[1:])
plt.show()
