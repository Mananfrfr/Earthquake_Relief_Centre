# pso_core.py
# Clean PSO core extracted from ReliefCentre_PSO.py — no plotting, no auto-execution

import math
from itertools import combinations



# -------------------------------
# Disaster Sites (copied as-is)
# -------------------------------
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

# -------------------------------
# Jaipur Bounding Box
# -------------------------------
LAT_MIN, LAT_MAX = 26.80, 27.05
LON_MIN, LON_MAX = 75.70, 76.00

# -------------------------------
# Simple Haversine Distance (no geopy)
# -------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# -------------------------------
# Cleaned Fitness Function (NO geopy!)
# -------------------------------
def fitness_function(particle_position, disaster_sites,
                     weight_distance=1.0,
                     weight_proximity=2.0,
                     weight_spread=1.0,
                     weight_coverage=2.0):
    """
    Composite fitness function with:
    1. Weighted distance to disaster sites
    2. Penalty if RC < 1.5 km from a site
    3. Incentive for high spread between RCs
    4. Coverage bonus (site within 5 km)
    5. Penalty if RC is >8 km away from ALL sites
    """

    # Convert flat particle list -> list of (lat, lon)
    relief_centers = [
        (particle_position[i], particle_position[i + 1])
        for i in range(0, len(particle_position), 2)
    ]

    # -------------------------------
    # 1. Weighted distance objective
    # -------------------------------
    weighted_distance = 0
    for site in disaster_sites:
        site_coord = (site["lat"], site["lon"])
        priority = site["priority"]

        distances = [
            haversine_km(site_coord[0], site_coord[1], rc[0], rc[1])
            for rc in relief_centers
        ]

        weighted_distance += min(distances) * (11 - priority)

    # -------------------------------
    # 2. Proximity penalty (<1.5 km)
    # -------------------------------
    penalty_proximity = 0
    for rc in relief_centers:
        for site in disaster_sites:
            d = haversine_km(rc[0], rc[1], site["lat"], site["lon"])
            if d < 1.5:
                penalty_proximity += (1.5 - d)

    # -------------------------------
    # 3. Spread between RCs
    # -------------------------------
    inter = []
    for (i, j) in combinations(relief_centers, 2):
        d = haversine_km(i[0], i[1], j[0], j[1])
        inter.append(d)

    if len(inter) == 0:
        avg_inter = 0
    else:
        avg_inter = sum(inter) / len(inter)

    # Spread penalty: higher spread = better = negative score
    spread_penalty = -avg_inter

    # -------------------------------
    # 4. Coverage bonus (<5 km)
    # -------------------------------
    covered_sites = 0
    for site in disaster_sites:
        if any(haversine_km(site["lat"], site["lon"], rc[0], rc[1]) <= 5.0
               for rc in relief_centers):
            covered_sites += 1

    coverage_bonus = -covered_sites

    # -------------------------------
    # 5. Too far (>8 km) penalty
    # -------------------------------
    too_far_penalty = 0
    for rc in relief_centers:
        d = min(
            haversine_km(rc[0], rc[1], s["lat"], s["lon"])
            for s in disaster_sites
        )
        if d > 8.0:
            too_far_penalty += (d - 8.0) ** 2

    # -------------------------------
    # Final fitness score
    # -------------------------------
    return (
        weight_distance * weighted_distance +
        weight_proximity * penalty_proximity +
        weight_spread * spread_penalty +
        weight_coverage * coverage_bonus +
        8.0 * too_far_penalty
    )
