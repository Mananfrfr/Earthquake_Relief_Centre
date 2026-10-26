import math

import numpy as np
import random
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from itertools import combinations
from matplotlib.animation import FuncAnimation





# Disaster site coordinates (lat, lon)
disaster_sites = [
    {"name": "Chandpole", "lat": 26.9330, "lon": 75.8235, "priority": 9},
    {"name": "Purani Basti", "lat": 26.9400, "lon": 75.8125, "priority": 8},
    {"name": "Sanganer", "lat": 26.8193, "lon": 75.8000, "priority": 10},
    {"name": "Vaishali Nagar", "lat": 26.9101, "lon": 75.7456, "priority": 6},
    {"name": "Shastri Nagar", "lat": 26.9503, "lon": 75.7904, "priority": 7},
    {"name": "Jagatpura", "lat": 26.8435, "lon": 75.8670, "priority": 10},
    {"name": "Jawahar Nagar", "lat": 26.9005, "lon": 75.8154, "priority": 6},
    {"name": "Malviya Nagar", "lat": 26.8595, "lon": 75.8069, "priority": 5}

]

# Jaipur bounding box (approximate)
LAT_MIN, LAT_MAX = 26.80, 27.05
LON_MIN, LON_MAX = 75.70, 76.00

NUM_RELIEF_CENTERS = 5
DIMENSIONS = NUM_RELIEF_CENTERS * 2

# PSO hyperparameters
SWARM_SIZE = 30
MAX_ITER = 50
W = 0.7  # Inertia weight
C1 = 1.5  # Cognitive (personal best)
C2 = 1.5  # Social (global best)









fig, ax = plt.subplots()
plt.ion()
particle_scatters = []
best_scatter = None
iteration_text = None




class Particle:
    def __init__(self):
        # Random initial position: [lat1, lon1, lat2, lon2, ..., lat5, lon5]
        self.position = [
            random.uniform(LAT_MIN, LAT_MAX) if i % 2 == 0 else random.uniform(LON_MIN, LON_MAX)
            for i in range(DIMENSIONS)
        ]
        self.velocity = [random.uniform(-0.01, 0.01) for _ in range(DIMENSIONS)]

        self.best_position = list(self.position)
        self.best_fitness = float('inf')

    def __str__(self):
        return f"Position: {self.position}\nFitness: {self.best_fitness:.2f}"

def update_plot(swarm, global_best_position,iteration):


    for i, particle in enumerate(swarm):
        rc_coords = [(particle.position[j + 1], particle.position[j]) for j in range(0, DIMENSIONS, 2)]
        lons, lats = zip(*rc_coords)
        particle_scatters[i].set_offsets(list(zip(lons, lats)))

    # Update best particle
    best_coords = [(global_best_position[j + 1], global_best_position[j]) for j in range(0, DIMENSIONS, 2)]
    best_lons, best_lats = zip(*best_coords)
    best_scatter.set_offsets(list(zip(best_lons, best_lats)))

    # Update iteration text
    iteration_text.set_text(f"Iteration: {iteration}")

    plt.draw()
    plt.pause(0.01)


def fitness_function(particle_position, disaster_sites,
                     weight_distance=1.0,
                     weight_proximity=2.0,
                     weight_spread=1.0,
                     weight_coverage=2.0):
    """
    Composite fitness function with tunable weights for:
    1. Weighted distance to disaster sites
    2. Penalty for RCs too close to disaster sites (< 1.5 km)
    3. Spread between RCs
    4. Coverage of disaster sites (within 5 km)

    Lower fitness is better.
    """

    relief_centers = [(particle_position[i], particle_position[i + 1]) for i in range(0, len(particle_position), 2)]

    # --- Objective 1: Weighted distance ---
    weighted_distance = 0
    for site in disaster_sites:
        site_coord = (site["lat"], site["lon"])
        priority = site["priority"]
        distances = [geodesic(site_coord, rc).km for rc in relief_centers]
        min_dist = min(distances)
        weighted_distance += min_dist * (11 - priority)

    # --- Objective 2: Penalty for too-close RCs (<1.5 km) ---
    penalty_proximity = 0
    for rc in relief_centers:
        for site in disaster_sites:
            site_coord = (site["lat"], site["lon"])
            dist = geodesic(rc, site_coord).km
            if dist < 1.5:
                penalty_proximity += (1.5 - dist)

    # --- Objective 3: Encourage spread (maximize RC-RC distances) ---
    inter_distances = [geodesic(rc1, rc2).km for rc1, rc2 in combinations(relief_centers, 2)]
    avg_inter_distance = sum(inter_distances) / len(inter_distances)
    spread_penalty = -avg_inter_distance

    # --- Objective 4: Coverage (maximize disaster sites within 5 km of any RC) ---
    covered_sites = 0
    for site in disaster_sites:
        site_coord = (site["lat"], site["lon"])
        if any(geodesic(site_coord, rc).km <= 5.0 for rc in relief_centers):
            covered_sites += 1
    coverage_bonus = -covered_sites  # More coverage → better

    # --- Objective 5: Penalize relief centers far from all disaster sites (>10 km) ---
    too_far_penalty = 0
    for rc in relief_centers:
        min_dist = min([geodesic(rc, (site["lat"], site["lon"])).km for site in disaster_sites])
        if min_dist > 8.0:
            too_far_penalty += (min_dist - 8.0) ** 2  # quadratic penalty


    # --- Weighted sum ---
    fitness = (
        weight_distance * weighted_distance +
        weight_proximity * penalty_proximity +
        weight_spread * spread_penalty +
        weight_coverage * coverage_bonus +
        8.0 * too_far_penalty
    )

    return fitness




"""def init_plot(disaster_sites):"""

ax.clear()

lats = [site["lat"] for site in disaster_sites]
lons = [site["lon"] for site in disaster_sites]
names = [site["name"] for site in disaster_sites]
priorities = [site1["priority"] for site1 in disaster_sites]
# Plot disaster sites (fixed)

scatter = ax.scatter(lons, lats, c=priorities, cmap="Reds", edgecolors="black", s=100, label="Disaster Site")
plt.colorbar(scatter, ax=ax, label="Priority")

# Add the initial of each disaster site name
for site in disaster_sites:
    initial = site["name"][0]  # or site["name"] for full name
    ax.text(site["lon"] + 0.002, site["lat"] + 0.002, initial, fontsize=8, color='darkred', weight='bold')
# plt.colorbar(scatter, label="Priority")

# Create a list of scatter plot handles for all particles


particle_scatters = [
    ax.scatter([], [], c="blue", marker="o", s=20) for _ in range(SWARM_SIZE)
]
# Best particle so far
best_scatter = ax.scatter([], [], c="yellow",marker='*',  s=150, label="Best RCs")

iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

ax.set_title("Real-Time PSO Optimization of Relief Centers")
ax.set_xlim(75.70, 76.00)
ax.set_ylim(26.80, 27.05)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.draw()
plt.pause(0.01)


def run_pso():

    # Initialize swarm
    swarm = [Particle() for _ in range(SWARM_SIZE)]


    # Evaluate initial fitness and find global best
    global_best_position = None
    global_best_fitness = float('inf')



    for particle in swarm:
        fitness = fitness_function(particle.position, disaster_sites,
                                   weight_distance=1.0,
                                   weight_proximity=5.0,
                                   weight_spread=0.5,
                                   weight_coverage=2.0)

        particle.best_fitness = fitness
        particle.best_position = particle.position[:]

        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = particle.position[:]

    # Main PSO loop
    for iter in range(MAX_ITER):
        for particle in swarm:
            for i in range(DIMENSIONS):
                r1 = random.random()
                r2 = random.random()

                cognitive = C1 * r1 * (particle.best_position[i] - particle.position[i])
                social = C2 * r2 * (global_best_position[i] - particle.position[i])
                particle.velocity[i] = W * particle.velocity[i] + cognitive + social

                particle.position[i] += particle.velocity[i]

                # Boundary constraints
                if i % 2 == 0:  # Latitude
                    particle.position[i] = max(min(particle.position[i], LAT_MAX), LAT_MIN)
                else:  # Longitude
                    particle.position[i] = max(min(particle.position[i], LON_MAX), LON_MIN)

            # Evaluate fitness
            fitness = fitness_function(particle.position, disaster_sites,
                                       weight_distance=1.0,
                                       weight_proximity=5.0,
                                       weight_spread=0.5,
                                       weight_coverage=2.0)

            # Update personal best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position[:]

            # Update global best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position[:]

        print(f"Iteration {iter+1}/{MAX_ITER} | Best Fitness: {global_best_fitness:.2f}")
        update_plot(swarm, global_best_position,iter)

    # Add RC labels after final positions are plotted
    for i in range(NUM_RELIEF_CENTERS):
        lat = global_best_position[2 * i]
        lon = global_best_position[2 * i + 1]
        ax.scatter(lon, lat, color='yellow', s=150, marker='*', edgecolors='black', zorder=5)  # 'X' marker with border
        ax.text(lon + 0.003, lat + 0.003, f"RC{i + 1}", fontsize=8, color='green')

    # Final result
    print("\nOptimal Relief Center Locations:")
    for i in range(NUM_RELIEF_CENTERS):
        lat = global_best_position[2 * i]
        lon = global_best_position[2 * i + 1]
        print(f"RC {i+1}: Lat = {lat:.5f}, Lon = {lon:.5f}")









if __name__ == "__main__":
    run_pso()
    plt.ioff()
    plt.show()