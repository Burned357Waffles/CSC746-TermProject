import math
import random
import time
import json
from dataclasses import dataclass, field

# Gravitational constant in SI units (m^3 kg^-1 s^-2)
G = 6.67430e-11
# 1 AU in meters
AU = 1.496e11
# 1 day in seconds
DAY = 24 * 60 * 60
# 1 solar mass in kg
SOLAR_MASS = 1.989e30

# change this to 2 to change the dimensions from 3D to 2D
DIM = 3
N = 0

# all objects will be stored in a list
objects = []

# timestep set to one hour
timestep = 60 * 60

# will end calculations after 24 timesteps
end_time = timestep * 24 * 365 * 5

@dataclass
class Object:
    mass: float
    velocity: list
    initial_position: list
    position: list = field(default_factory=list)

    def __post_init__(self):
        self.position.append([p * AU for p in self.initial_position])  # Convert AU to meters
        self.velocity = [v * AU / DAY for v in self.velocity]  # Convert AU/day to meters/second

def update_forces(force, obj):
    pos = [0.0] * DIM
    for dim in range(DIM):
        obj.velocity[dim] += (force[dim] / obj.mass) * timestep
        pos[dim] = obj.position[-1][dim] + obj.velocity[dim] * timestep
    obj.position.append(pos)

def compute_forces(i):
    total_force = [0.0] * DIM

    for j in objects:
        if j == i:
            continue

        dx = []
        r = 0

        for index in range(DIM):
            dx.append(j.position[-1][index] - i.position[-1][index])

        for index in range(DIM):
            r += dx[index] ** 2

        r_norm = math.sqrt(r)
        if r_norm == 0:
            continue

        for dim in range(DIM):
            force = (G * i.mass * j.mass * dx[dim]) / (r_norm ** 3)
            total_force[dim] += force

    return total_force

def do_nBody_calculation():
    for t in range(0, end_time, timestep):
        forces = []
        for obj in objects:
            force = compute_forces(obj)
            update_forces(force, obj)
            forces.append(force)

        #print(f"Time: {t} seconds")

def init_objects():
    objects.append(Object(SOLAR_MASS, [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]))  # Sol
    objects.append(Object(SOLAR_MASS, [1.0e-6, 0.0, 0.0], [1.0, 1.0, 1.0]))

    for n in range(N):
        mass = random.uniform(1.0e-6, 1.0) * SOLAR_MASS  # Convert solar masses to kg
        velocity = [random.uniform(-0.1, 0.1) for _ in range(DIM)]
        initial_position = list(random.uniform(-1.0, 1.0) for _ in range(DIM))

        objects.append(Object(mass, velocity, initial_position))

def save_positions(objects, filename):
    positions = [obj.position for obj in objects]
    with open(filename, 'w') as f:
        json.dump(positions, f)

def main():
    init_objects()
    start_timer = time.time()
    do_nBody_calculation()
    end_timer = time.time()

    print(f"Elapsed time: {end_timer - start_timer}")

    save_positions(objects, 'positions.json')

if __name__ == '__main__':
    main()