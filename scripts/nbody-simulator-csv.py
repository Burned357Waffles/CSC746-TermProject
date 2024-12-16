import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

colorful = True

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

# Set background color to black
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Set grid color to white
ax.xaxis._axinfo['grid'].update(color='white')
ax.yaxis._axinfo['grid'].update(color='white')
ax.zaxis._axinfo['grid'].update(color='white')

# Set the pane colors to black
#ax.xaxis.set_pane_color((0, 0, 0, 1))
#ax.yaxis.set_pane_color((0, 0, 0, 1))
#ax.zaxis.set_pane_color((0, 0, 0, 1))

# Identify the object with the greatest mass
max_mass_index = df['m'].idxmax()
max_mass_body = df.loc[max_mass_index, 'body_num']

unique_bodies = b.unique()
colors = plt.cm.jet(np.linspace(0, 1, len(unique_bodies)))[::-1]
# Plotting the first and last positions with dots and keeping the line
for body in b.unique():
    body_mask = (b == body)
    x_body = x[body_mask]
    y_body = y[body_mask]
    z_body = z[body_mask]
    m_body = m[body_mask]

    if colorful:
        color = colors[body]
    else:
        if body == max_mass_body:
            color = 'red'
        else:
            color = 'blue'

    # Plot the first and last positions with dots
    ax.scatter(x_body.iloc[[0]], y_body.iloc[[0]], z_body.iloc[[0]], s=m_body.iloc[[0]], color=color)
    ax.scatter(x_body.iloc[[-1]], y_body.iloc[[-1]], z_body.iloc[[-1]], s=m_body.iloc[[-1]], color=color, alpha=0.5)

    # Plot the line for the entire trajectory
    ax.plot3D(x_body, y_body, z_body, color=color)

max_val = max(x.max(), y.max(), z.max())
min_val = min(x.min(), y.min(), z.min())

ax.set_xlim([min_val, max_val])
ax.set_ylim([min_val, max_val])
ax.set_zlim([min_val, max_val])

ax.set_xlabel('X axis', color='white')
ax.set_ylabel('Y axis', color='white')
ax.set_zlabel('Z axis', color='white')

ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')

ax.view_init(elev=10., azim=45)
plt.show()