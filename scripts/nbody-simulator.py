import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filename = 'positions.json'
with open(filename, 'r') as f:
    positions = json.load(f)

fig = plt.figure(figsize=(20, 10))
ax = plt.axes(projection='3d')

for obj_positions in positions:
    x, y, z = zip(*obj_positions)
    ax.scatter(x, y, z, s=5)
    ax.plot3D(x, y, z)

ax.view_init(elev=10., azim=45)
plt.show()