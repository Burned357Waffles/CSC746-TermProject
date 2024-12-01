import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Read the CSV file
filename = '../data/positions.csv'
df = pd.read_csv(filename)

# Debugging: Print the DataFrame and its length
print(df)
print(len(df))

# Assuming df is your DataFrame
b = df['body_num']
x = df['x']
y = df['y']
z = df['z']

# Get unique bodies
unique_bodies = b.unique()

# Create a colormap
colors = plt.cm.jet(np.linspace(0, 1, len(unique_bodies)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the plot
scatters = []
lines = []
for i, body in enumerate(unique_bodies):
    body_mask = (b == body)
    scatter = ax.scatter([], [], [], s=5, color=colors[i])
    line, = ax.plot3D([], [], [], color=colors[i])
    scatters.append(scatter)
    lines.append(line)

def init():
    for scatter, line in zip(scatters, lines):
        scatter._offsets3d = ([], [], [])
        line.set_data([], [])
        line.set_3d_properties([])
    return scatters + lines

def update(frame):
    for i, body in enumerate(unique_bodies):
        body_mask = (b == body)
        scatters[i]._offsets3d = (x[body_mask][:frame], y[body_mask][:frame], z[body_mask][:frame])
        lines[i].set_data(x[body_mask][:frame], y[body_mask][:frame])
        lines[i].set_3d_properties(z[body_mask][:frame])
    return scatters + lines

ani = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True)

ax.view_init(elev=10., azim=45)
plt.show()