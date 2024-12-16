import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

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

unique_bodies = b.unique()
colors = plt.cm.jet(np.linspace(0, 1, len(unique_bodies)))[::-1]

# Initialize scatter and line objects
scatters = []
lines = []
for body in unique_bodies:
    scatter = ax.scatter([], [], [], s=[], color=colors[body])
    line, = ax.plot3D([], [], [], color=colors[body])
    scatters.append(scatter)
    lines.append(line)

max_val = max(x.max(), y.max(), z.max())
min_val = min(x.min(), y.min(), z.min())

zooming_max = max_val * 0.7
zooming_min = min_val * 0.7

ax.set_xlim([zooming_min, zooming_max])
ax.set_ylim([zooming_min, zooming_max])
ax.set_zlim([zooming_min, zooming_max])

"""
ax.set_xlim([min_val, max_val])
ax.set_ylim([min_val, max_val])
ax.set_zlim([min_val, max_val])
"""
ax.set_xlabel('X axis', color='white')
ax.set_ylabel('Y axis', color='white')
ax.set_zlabel('Z axis', color='white')

ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')

ax.view_init(elev=10., azim=45)

def update(frame):
    for i, body in enumerate(unique_bodies):
        body_mask = (b == body)
        x_body = x[body_mask]
        y_body = y[body_mask]
        z_body = z[body_mask]
        m_body = m[body_mask]

        scatters[i]._offsets3d = (x_body.iloc[frame:frame+1], y_body.iloc[frame:frame+1], z_body.iloc[frame:frame+1])
        scatters[i].set_sizes([m_body.iloc[frame]])
        lines[i].set_data(x_body.iloc[:frame+1], y_body.iloc[:frame+1])
        lines[i].set_3d_properties(z_body.iloc[:frame+1])

    return scatters + lines

# TO WRITE TO GIF UNCOMMENT THIS LINE
#frames = min(len(df) // len(unique_bodies), 1000)

# TO WRITE TO GIF COMMENT THIS LINE
frames = len(df) // len(unique_bodies)

ani = FuncAnimation(fig, update, frames=frames, interval=10, blit=False)

# TO WRITE TO GIF UNCOMMENT THIS LINE
#ani.save('nbody_animation-3-bodies.gif', writer=PillowWriter(fps=30))

# TO WRITE TO GIF COMMENT THIS LINE
plt.show()