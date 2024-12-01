import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
filename = '../data/positions.csv'
df = pd.read_csv(filename)

# Print the columns to debug
print(df.columns)

fig = plt.figure(figsize=(20, 10))
ax = plt.axes(projection='3d')

# Extract x, y, z columns
b = df['body_num']
x = df['x']
y = df['y']
z = df['z']

# Get unique bodies
unique_bodies = b.unique()

# Create a colormap
colors = plt.cm.jet(np.linspace(0, 1, len(unique_bodies)))

# Plot the data
for i, body in enumerate(unique_bodies):
    body_mask = (b == body)
    ax.scatter(x[body_mask], y[body_mask], z[body_mask], s=5, color=colors[i])
    ax.plot3D(x[body_mask], y[body_mask], color=colors[i])

ax.view_init(elev=10., azim=45)
plt.show()