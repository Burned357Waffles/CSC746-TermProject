import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filename = 'data/positions.csv'
df = pd.read_csv(filename, skiprows=1, header=None, names=['body_num', 'm', 'vx', 'vy', 'vz', 'x', 'y', 'z'])

# Convert 'm' column to numeric, forcing errors to NaN
df['m'] = pd.to_numeric(df['m'], errors='coerce')

# Normalize 'm' to the range 0 to 1
m_min = df['m'].min()
m_max = df['m'].max()

if m_min == m_max:
    df['m_normalized'] = 1  # Default value when all masses are the same
else:
    df['m_normalized'] = (df['m'] - m_min) / (m_max - m_min)

# Scale normalized 'm' to the range 1 to 50
df['m_scaled'] = df['m_normalized'] * 99 + 15

# Extract columns
b = df['body_num']
m = df['m_scaled']
vx = df['vx']
vy = df['vy']
vz = df['vz']
x = df['x']
y = df['y']
z = df['z']

fig = plt.figure(figsize=(20, 10))
ax = plt.axes(projection='3d')

# Identify the object with the greatest mass
max_mass_index = df['m'].idxmax()
max_mass_body = df.loc[max_mass_index, 'body_num']

# Plotting the first and last positions with dots and keeping the line
for body in b.unique():
    body_mask = (b == body)
    x_body = x[body_mask]
    y_body = y[body_mask]
    z_body = z[body_mask]
    m_body = m[body_mask]

    if body == max_mass_body:
        color = 'red'
    else:
        color = 'blue'

    # Plot the first and last positions with dots
    ax.scatter(x_body.iloc[[0, -1]], y_body.iloc[[0, -1]], z_body.iloc[[0, -1]], s=m_body.iloc[[0, -1]], color=color)

    # Plot the line for the entire trajectory
    ax.plot3D(x_body, y_body, z_body, color=color)

max_val = max(x.max(), y.max(), z.max())
min_val = min(x.min(), y.min(), z.min())

ax.set_xlim([min_val, max_val])
ax.set_ylim([min_val, max_val])
ax.set_zlim([min_val, max_val])

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

ax.view_init(elev=10., azim=45)
plt.show()