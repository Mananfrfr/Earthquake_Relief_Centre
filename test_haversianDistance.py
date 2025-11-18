"""from ReliefCentre_PSO import haversine_distance


# Test with two known points
lat1, lon1 = 26.926, 75.812  # Chandpole Bazar
lat2, lon2 = 26.915, 75.820  # Johari Bazaar

distance_km = haversine_distance(lat1, lon1, lat2, lon2)
print(f"Distance: {distance_km:.2f} km")"""

import matplotlib.pyplot as plt
import numpy as np
import time

plt.ion()  # Interactive mode ON

fig, ax = plt.subplots()
sc = ax.scatter([], [], c='blue')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

for i in range(50):
    x = np.random.rand(5) * 10
    y = np.random.rand(5) * 10
    sc.set_offsets(np.column_stack((x, y)))
    text.set_text(f"Iteration: {i+1}")
    plt.draw()
    plt.pause(0.1)

plt.ioff()
plt.show()
