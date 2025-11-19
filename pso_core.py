# pso_core.py
# PSO fitness & helpers for Earthquake Relief app
# - Supports per-RC supply vector
# - Computes dynamic bounding box from disaster sites and applies strong soft penalties
# - Uses haversine_km for distances

import math
from itertools import combinations

# -------------------------------
# Example/default disaster sites (app will override)
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
# Utilities
# -------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def get_bounds_from_sites(disaster_sites, pad_lat=0.005, pad_lon=0.005):
    """
    Compute a tight bounding box around the currently marked disaster_sites.
    Returns (lat_min, lat_max, lon_min, lon_max).
    Adds small padding to avoid edge cases.
    """
    if not disaster_sites:
        # fallback to a reasonable Jaipur-ish bounding if nothing provided
        return 26.80, 27.05, 75.70, 76.00
    lats = [s["lat"] for s in disaster_sites]
    lons = [s["lon"] for s in disaster_sites]
    return min(lats) - pad_lat, max(lats) + pad_lat, min(lons) - pad_lon, max(lons) + pad_lon

# -------------------------------
# Fitness function (per-RC supplies + bbox enforcement)
# -------------------------------
def fitness_function(particle_position, disaster_sites,
                     supply_vector=None,
                     weight_distance=1.0,
                     weight_proximity=2.0,
                     weight_spread=1.0,
                     weight_coverage=2.0,
                     bbox_penalty_scale=5000.0):
    """
    Composite fitness function.
    - particle_position: flat list [lat0, lon0, lat1, lon1, ...]
    - disaster_sites: list of dicts with lat/lon and priority
    - supply_vector: list of supply values per RC (can be None)
    - bbox_penalty_scale: scale for penalty when RC goes outside dynamic bbox
    Returns scalar: lower is better.
    """

    # parse relief centers
    relief_centers = [
        (particle_position[i], particle_position[i + 1])
        for i in range(0, len(particle_position), 2)
    ]
    n_rc = len(relief_centers)

    # normalize/validate supply_vector
    if supply_vector is None:
        supply_vector = [1000.0] * n_rc
    else:
        sv = list(supply_vector)
        if len(sv) < n_rc:
            pad_val = float(sv[-1]) if sv else 1000.0
            sv = sv + [max(1.0, pad_val)] * (n_rc - len(sv))
        if len(sv) > n_rc:
            sv = sv[:n_rc]
        supply_vector = [float(s) if (s and float(s) > 0.0) else 1.0 for s in sv]

    # -------------------------------
    # 1) Assign each site to its nearest RC (for contribution calculation)
    # -------------------------------
    site_assignments = []
    for site in disaster_sites:
        site_lat, site_lon = site["lat"], site["lon"]
        dists = [haversine_km(site_lat, site_lon, rc[0], rc[1]) for rc in relief_centers]
        # find index of minimum distance
        if len(dists) == 0:
            continue
        min_idx = int(min(range(len(dists)), key=lambda i: dists[i]))
        site_assignments.append((min_idx, dists[min_idx], site.get("priority", 5)))

    # accumulate per-RC weighted distance (priority-aware)
    per_rc_weighted_distance = [0.0 for _ in range(n_rc)]
    for assigned_idx, dist, priority in site_assignments:
        multiplier = (11 - priority)  # higher priority -> smaller multiplier
        per_rc_weighted_distance[assigned_idx] += dist * multiplier

    # distance term: sum(weighted_distance_i / supply_i)
    distance_term = 0.0
    for i in range(n_rc):
        distance_term += (per_rc_weighted_distance[i] / supply_vector[i])

    # -------------------------------
    # 2) Proximity penalty (<1.5 km)
    # -------------------------------
    penalty_proximity = 0.0
    for rc in relief_centers:
        for site in disaster_sites:
            d = haversine_km(rc[0], rc[1], site["lat"], site["lon"])
            if d < 1.5:
                penalty_proximity += (1.5 - d)

    # -------------------------------
    # 3) Spread between RCs (encourage spread)
    # -------------------------------
    inter = []
    for (i, j) in combinations(relief_centers, 2):
        d = haversine_km(i[0], i[1], j[0], j[1])
        inter.append(d)
    avg_inter = sum(inter) / len(inter) if inter else 0.0
    spread_penalty = -avg_inter  # more spread reduces fitness

    # -------------------------------
    # 4) Coverage bonus (<5 km)
    # -------------------------------
    covered_sites = 0
    for site in disaster_sites:
        if any(haversine_km(site["lat"], site["lon"], rc[0], rc[1]) <= 5.0 for rc in relief_centers):
            covered_sites += 1
    coverage_bonus = -covered_sites

    # -------------------------------
    # 5) Too-far penalty (>8 km)
    # -------------------------------
    too_far_penalty = 0.0
    for rc in relief_centers:
        dmin = min(haversine_km(rc[0], rc[1], s["lat"], s["lon"]) for s in disaster_sites)
        if dmin > 8.0:
            too_far_penalty += (dmin - 8.0) ** 2

    # -------------------------------
    # 6) Dynamic bounding-box penalty (strong)
    # -------------------------------
    lat_min, lat_max, lon_min, lon_max = get_bounds_from_sites(disaster_sites)
    bbox_penalty = 0.0
    for (lat, lon) in relief_centers:
        out_lat = 0.0
        out_lon = 0.0
        if lat < lat_min:
            out_lat = (lat_min - lat)
        elif lat > lat_max:
            out_lat = (lat - lat_max)
        if lon < lon_min:
            out_lon = (lon_min - lon)
        elif lon > lon_max:
            out_lon = (lon - lon_max)
        # square the offset to punish further excursions more heavily
        bbox_penalty += (out_lat ** 2 + out_lon ** 2) * bbox_penalty_scale

    # -------------------------------
    # Final aggregated fitness
    # -------------------------------
    final_score = (
        weight_distance * distance_term +
        weight_proximity * penalty_proximity +
        weight_spread * spread_penalty +
        weight_coverage * coverage_bonus +
        8.0 * too_far_penalty +
        bbox_penalty
    )

    return final_score
