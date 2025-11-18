# Resource Allocation Optimization for Disaster Relief
# =================================================
# This program optimizes the allocation of resources from relief centers to disaster sites
# using three different multi-objective evolutionary algorithms (NSGA-II, MOEA/D, and SPEA2).
# The objectives are:
# 1. Minimize total delivery time
# 2. Minimize total distance traveled
# 3. Maximize priority fulfillment based on severity and people affected

# Import necessary libraries
import networkx as nx
import random
import numpy as np
import osmnx as ox
import openrouteservice
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.util.ref_dirs import get_reference_directions
from mpl_toolkits.mplot3d import Axes3D

# Relief Centers data
relief_centers = [
    {"name": "RC0", "latitude":26.909313, "longitude":75.780869, "area": "Civil Lines"},
    {"name": "RC1", "latitude": 26.857159, "longitude": 75.815384, "area": "Malviya Nagar"},
    {"name": "RC2", "latitude": 26.926789, "longitude": 75.792982, "area": "Bani Park"},
    {"name": "RC3", "latitude": 26.868720, "longitude": 75.783923, "area": "Gopalpura"},
    {"name": "RC4", "latitude":26.863092, "longitude": 75.754264, "area": "Mansarovar"},
]

# Disaster Sites data
disaster_sites = [
    {"name": "DS0", "latitude": 26.9260, "longitude": 75.7885, "area": "MI Road"},
    {"name": "DS1", "latitude": 26.9121, "longitude": 75.8012, "area": "Lalkothi"},
    {"name": "DS2", "latitude": 26.8948, "longitude": 75.8239, "area": "Jawahar Nagar"},
    {"name": "DS3", "latitude": 26.9354, "longitude": 75.8042, "area": "Chandpole"},
    {"name": "DS4", "latitude": 26.9157, "longitude": 75.7485, "area": "Kanakpura"},
    {"name": "DS5", "latitude": 26.891067, "longitude": 75.840884, "area": "Galtaji"},
    {"name": "DS6", "latitude": 26.935095, "longitude": 75.833358, "area": "Adarsh Nagar"},
    {"name": "DS7", "latitude": 26.898527, "longitude": 75.761034, "area": "Shyam Nagar"},
]

# Populate additional fields with random values
for ds in disaster_sites:
    ds["people_affected"] = random.randint(50, 500)  # Random number of people affected
    ds["severity_level"] = random.randint(1, 10)  # Severity level from 1 to 10
    ds["time_since_last_response"] = random.randint(1, 24)  # Time in hours since last response
    ds["demand"] = random.randint(100, 1000)  # Random demand for resources

for rc in relief_centers:
    rc["supply"] = random.randint(500, 2000)  # Random supply of resources

# Print updated relief centers and disaster sites for verification
print("\nUpdated Relief Centers with Supply:")
for rc in relief_centers:
    print(rc)

print("\nUpdated Disaster Sites with Demand:")
for ds in disaster_sites:
    print(ds)

# Assign Priority Scores to Disaster Sites
# priority[j] = people_affected[j] * severity[j] / normalization
max_people_affected = max(ds["people_affected"] for ds in disaster_sites)
max_severity_level = max(ds["severity_level"] for ds in disaster_sites)
normalization = max_people_affected * max_severity_level

for ds in disaster_sites:
    ds["priority_score"] = (ds["people_affected"] * ds["severity_level"]) / normalization

# Print Priority Scores for verification
print("\nPriority Scores for Disaster Sites:")
for ds in disaster_sites:
    print(f"{ds['name']} (Area: {ds['area']}): Priority Score = {ds['priority_score']:.2f}")

# Haversine formula to calculate distance between two lat-long points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lat2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Initialize OpenRouteService client
ORS_API_KEY = ""  # Replace with your OpenRouteService API key : 5b3ce3597851110001cf62487c4f201b3ce640f684117feac321399b
client = openrouteservice.Client(key=ORS_API_KEY)

# Create a graph using osmnx
graph = ox.graph_from_point((26.9124, 75.7873), dist=5000, network_type='drive')  # 5 km radius

# Function to calculate distance and time using OpenRouteService
def get_distance_time(lat1, lon1, lat2, lon2):
    try:
        coords = ((lon1, lat1), (lon2, lat2))
        route = client.directions(coords, profile='driving-car', format='json')
        distance = route['routes'][0]['summary']['distance'] / 1000  # Convert meters to kilometers
        duration = route['routes'][0]['summary']['duration'] / 60  # Convert seconds to minutes
        return distance, duration
    except Exception as e:
        print(f"Error fetching route: {e}")
        return float('inf'), float('inf')

# Function to calculate distance and time using Haversine formula
def haversine_distance_time(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lat2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c  # Distance in kilometers
    urban_speed_kmh = 30  # Urban travel speed in km/h
    time = (distance / urban_speed_kmh) * 60  # Time in minutes
    return distance, time

# Compute Distance and Time Matrices
distance_matrix = []
time_matrix = []

for rc in relief_centers:
    distances = []
    times = []
    for ds in disaster_sites:
        if ORS_API_KEY.strip():  # Use OpenRouteService if API key is provided
            distance, time = get_distance_time(rc["latitude"], rc["longitude"], ds["latitude"], ds["longitude"])
        else:  # Fallback to Haversine formula
            distance, time = haversine_distance_time(rc["latitude"], rc["longitude"], ds["latitude"], ds["longitude"])
        distances.append(distance)
        times.append(time)
    distance_matrix.append(distances)
    time_matrix.append(times)

# Print Distance Matrix
print("\nDistance Matrix (in km):")
for i, row in enumerate(distance_matrix):
    print(f"{relief_centers[i]['name']}: {row}")

# Print Time Matrix
print("\nTime Matrix (in minutes):")
for i, row in enumerate(time_matrix):
    print(f"{relief_centers[i]['name']}: {row}")

# Extract coordinates for plotting
relief_latitudes = [rc["latitude"] for rc in relief_centers]
relief_longitudes = [rc["longitude"] for rc in relief_centers]
relief_labels = [rc["name"] for rc in relief_centers]

disaster_latitudes = [ds["latitude"] for ds in disaster_sites]
disaster_longitudes = [ds["longitude"] for ds in disaster_sites]
disaster_labels = [ds["name"] for ds in disaster_sites]

# Plot the map
plt.figure(figsize=(10, 8))
plt.scatter(relief_longitudes, relief_latitudes, c='blue', label='Relief Centers', s=100, marker='o')
plt.scatter(disaster_longitudes, disaster_latitudes, c='red', label='Disaster Sites', s=100, marker='x')

# Add labels to points
for i, label in enumerate(relief_labels):
    plt.text(relief_longitudes[i] + 0.002, relief_latitudes[i], label, fontsize=9, color='blue')
for i, label in enumerate(disaster_labels):
    plt.text(disaster_longitudes[i] + 0.002, disaster_latitudes[i], label, fontsize=9, color='red')

# Add title and legend
plt.title("Relief Centers and Disaster Sites", fontsize=14)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

# ===================================================
# PROBLEM DEFINITION
# ===================================================
print("\n" + "="*50)
print("DISASTER RELIEF RESOURCE ALLOCATION PROBLEM")
print("="*50)

# Define the NSGA-II problem
class ResourceAllocationProblem(Problem):
    def __init__(self, relief_centers, disaster_sites, distance_matrix, time_matrix):
        self.relief_centers = relief_centers
        self.disaster_sites = disaster_sites
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix

        # Number of variables: one for each relief center-disaster site pair
        n_var = len(relief_centers) * len(disaster_sites)
        xl = np.zeros(n_var)  # Lower bounds (no negative allocations)
        xu = np.array([min(rc["supply"], ds["demand"]) for rc in relief_centers for ds in disaster_sites])  # Upper bounds

        # Number of objectives:
        # 1. Minimize total delivery time
        # 2. Minimize total distance
        # 3. Maximize priority fulfillment
        n_obj = 3

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        n_solutions = x.shape[0]
        f1 = np.zeros(n_solutions)  # Total delivery time
        f2 = np.zeros(n_solutions)  # Total distance
        f3 = np.zeros(n_solutions)  # Negative priority fulfillment (to maximize)

        for i in range(n_solutions):
            # Round allocations to integers
            allocation = np.rint(x[i].reshape(len(self.relief_centers), len(self.disaster_sites))).astype(int)
            total_time = 0
            total_distance = 0
            priority_fulfillment = 0

            for rc_idx, rc in enumerate(self.relief_centers):
                for ds_idx, ds in enumerate(self.disaster_sites):
                    allocated = allocation[rc_idx, ds_idx]
                    if allocated > 0:
                        total_time += self.time_matrix[rc_idx][ds_idx] * allocated
                        total_distance += self.distance_matrix[rc_idx][ds_idx] * allocated
                        priority_fulfillment += allocated * ds["priority_score"]

            f1[i] = total_time
            f2[i] = total_distance
            f3[i] = -priority_fulfillment  # Negative because we want to maximize

        out["F"] = np.column_stack([f1, f2, f3])

# Create the problem instance
problem = ResourceAllocationProblem(relief_centers, disaster_sites, distance_matrix, time_matrix)

# Function to extract the allocation from a solution vector
def get_allocation(solution_vector):
    allocation = solution_vector.reshape(len(relief_centers), len(disaster_sites))
    allocation_dict = {}
    for rc_idx, rc in enumerate(relief_centers):
        for ds_idx, ds in enumerate(disaster_sites):
            allocated = allocation[rc_idx, ds_idx]
            if allocated > 0:
                allocation_dict[(rc["name"], ds["name"])] = allocated
    return allocation_dict

# Function to print allocation details
def print_allocation_details(allocation, solution):
    total_time = solution[0]
    total_distance = solution[1]
    priority_fulfillment = -solution[2]  # Convert back to positive
    
    print(f"   Total Delivery Time: {total_time:.2f} minutes")
    print(f"   Total Distance: {total_distance:.2f} km")
    print(f"   Priority Fulfillment: {priority_fulfillment:.4f}")
    
    print("\n   Resource Allocation:")
    for (rc, ds), allocated in allocation.items():
        print(f"     {rc} -> {ds}: {allocated} units")

# Function to visualize a Pareto front
def visualize_pareto_front(F, X, title, algorithm_name, color='blue'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all solutions on the Pareto front
    ax.scatter(F[:, 0], F[:, 1], -F[:, 2], c=color, s=30, alpha=0.6, label='Pareto Front')
    
    # Highlight the best solution (minimum total time)
    min_time_idx = np.argmin(F[:, 0])
    min_time_solution = F[min_time_idx]
    ax.scatter([min_time_solution[0]], [min_time_solution[1]], [-min_time_solution[2]], 
               c='red', s=100, label='Min Time Solution')
    
    # Set axis labels
    ax.set_xlabel('Total Time (minutes)', fontsize=12)
    ax.set_ylabel('Total Distance (km)', fontsize=12)
    ax.set_zlabel('Priority Fulfillment', fontsize=12)
    
    # Add title and legend
    ax.set_title(f'Pareto Front: {title}', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Return the minimum time solution for comparison
    return min_time_idx, min_time_solution

# ===================================================
# ALGORITHM 1: NSGA-II OPTIMIZATION
# ===================================================
print("\n" + "="*50)
print("ALGORITHM 1: NSGA-II OPTIMIZATION")
print("="*50)

# Create the NSGA-II algorithm
algorithm = NSGA2(
    pop_size=100,
    n_offsprings=100,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# Run the optimization
print("\nRunning NSGA-II optimization...")
result_nsga2 = minimize(problem, algorithm, ('n_gen', 100), seed=42, verbose=True)

# Print the results
print("\n=== NSGA-II Optimization Results ===")
print(f"Execution Time: {result_nsga2.exec_time:.2f} seconds")
print(f"Number of evaluations: {result_nsga2.algorithm.evaluator.n_eval}")
print(f"Number of solutions on Pareto front: {len(result_nsga2.F)}")

# Extract the best solution (minimum total time)
min_time_idx_nsga2, min_time_solution_nsga2 = visualize_pareto_front(
    result_nsga2.F, result_nsga2.X, "NSGA-II Optimization", "NSGA-II", color='blue')
min_time_allocation_nsga2 = get_allocation(result_nsga2.X[min_time_idx_nsga2])

print("\nBest NSGA-II Solution (Minimum Total Time):")
print_allocation_details(min_time_allocation_nsga2, min_time_solution_nsga2)

# ===================================================
# ALGORITHM 2: MOEA/D OPTIMIZATION
# ===================================================
print("\n" + "="*50)
print("ALGORITHM 2: MOEA/D OPTIMIZATION")
print("="*50)

# Create reference directions for MOEA/D
ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_partitions=12)

# Create the MOEA/D algorithm
moead_algorithm = MOEAD(
    ref_dirs=ref_dirs,
    decomposition=Tchebicheff(),
    n_neighbors=15,
    prob_neighbor_mating=0.7,
    seed=42,
    verbose=True
)

# Run the MOEA/D optimization
print("\nRunning MOEA/D optimization...")
result_moead = minimize(problem, moead_algorithm, ('n_gen', 100), seed=42, verbose=True)

# Print the MOEA/D results
print("\n=== MOEA/D Optimization Results ===")
print(f"Execution Time: {result_moead.exec_time:.2f} seconds")
print(f"Number of evaluations: {result_moead.algorithm.evaluator.n_eval}")
print(f"Number of solutions on Pareto front: {len(result_moead.F)}")

# Extract the best solution (minimum total time)
min_time_idx_moead, min_time_solution_moead = visualize_pareto_front(
    result_moead.F, result_moead.X, "MOEA/D Optimization", "MOEA/D", color='green')
min_time_allocation_moead = get_allocation(result_moead.X[min_time_idx_moead])

print("\nBest MOEA/D Solution (Minimum Total Time):")
print_allocation_details(min_time_allocation_moead, min_time_solution_moead)

# ===================================================
# ALGORITHM 3: SPEA2 OPTIMIZATION
# ===================================================
print("\n" + "="*50)
print("ALGORITHM 3: SPEA2 OPTIMIZATION")
print("="*50)

# Create the SPEA2 algorithm
spea2_algorithm = SPEA2(
    pop_size=100,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# Run the SPEA2 optimization
print("\nRunning SPEA2 optimization...")
result_spea2 = minimize(problem, spea2_algorithm, ('n_gen', 100), seed=42, verbose=True)

# Print the SPEA2 results
print("\n=== SPEA2 Optimization Results ===")
print(f"Execution Time: {result_spea2.exec_time:.2f} seconds")
print(f"Number of evaluations: {result_spea2.algorithm.evaluator.n_eval}")
print(f"Number of solutions on Pareto front: {len(result_spea2.F)}")

# Extract the best solution (minimum total time)
min_time_idx_spea2, min_time_solution_spea2 = visualize_pareto_front(
    result_spea2.F, result_spea2.X, "SPEA2 Optimization", "SPEA2", color='purple')
min_time_allocation_spea2 = get_allocation(result_spea2.X[min_time_idx_spea2])

print("\nBest SPEA2 Solution (Minimum Total Time):")
print_allocation_details(min_time_allocation_spea2, min_time_solution_spea2)

# ===================================================
# ALGORITHM COMPARISON
# ===================================================
print("\n" + "="*50)
print("ALGORITHM COMPARISON")
print("="*50)

# Create performance comparison dataframe
performance_df = pd.DataFrame({
    'Algorithm': ['NSGA-II', 'MOEA/D', 'SPEA2'],
    'Execution Time (s)': [result_nsga2.exec_time, result_moead.exec_time, result_spea2.exec_time],
    'Pareto Front Size': [len(result_nsga2.F), len(result_moead.F), len(result_spea2.F)],
    'Min Delivery Time': [min_time_solution_nsga2[0], min_time_solution_moead[0], min_time_solution_spea2[0]],
    'Distance for Min Time': [min_time_solution_nsga2[1], min_time_solution_moead[1], min_time_solution_spea2[1]],
    'Priority for Min Time': [-min_time_solution_nsga2[2], -min_time_solution_moead[2], -min_time_solution_spea2[2]]
})

print("\nPerformance Comparison:")
print(performance_df)

# Find the best algorithm in terms of minimum delivery time
best_time_algo = performance_df.loc[performance_df['Min Delivery Time'].idxmin()]['Algorithm']
best_time_value = performance_df['Min Delivery Time'].min()

# Find the best algorithm in terms of maximum priority fulfillment
best_priority_algo = performance_df.loc[performance_df['Priority for Min Time'].idxmax()]['Algorithm']
best_priority_value = performance_df['Priority for Min Time'].max()

# Determine overall best algorithm
print("\nBest Algorithm Analysis:")
print(f"- Best for Minimum Delivery Time: {best_time_algo} ({best_time_value:.2f} minutes)")
print(f"- Best for Maximum Priority Fulfillment: {best_priority_algo} ({best_priority_value:.4f})")

# Visualize the Pareto fronts of all three algorithms together
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot Pareto fronts
ax.scatter(result_nsga2.F[:, 0], result_nsga2.F[:, 1], -result_nsga2.F[:, 2], c='blue', s=30, alpha=0.6, label='NSGA-II')
ax.scatter(result_moead.F[:, 0], result_moead.F[:, 1], -result_moead.F[:, 2], c='green', s=30, alpha=0.6, label='MOEA/D')
ax.scatter(result_spea2.F[:, 0], result_spea2.F[:, 1], -result_spea2.F[:, 2], c='purple', s=30, alpha=0.6, label='SPEA2')

# Highlight the best solutions (minimum total time)
ax.scatter([min_time_solution_nsga2[0]], [min_time_solution_nsga2[1]], [-min_time_solution_nsga2[2]], 
           c='red', s=100, label='NSGA-II Best')
ax.scatter([min_time_solution_moead[0]], [min_time_solution_moead[1]], [-min_time_solution_moead[2]], 
           c='orange', s=100, label='MOEA/D Best')
ax.scatter([min_time_solution_spea2[0]], [min_time_solution_spea2[1]], [-min_time_solution_spea2[2]], 
           c='magenta', s=100, label='SPEA2 Best')

# Set axis labels
ax.set_xlabel('Total Time (minutes)', fontsize=12)
ax.set_ylabel('Total Distance (km)', fontsize=12)
ax.set_zlabel('Priority Fulfillment', fontsize=12)

# Add title and legend
ax.set_title('Pareto Front Comparison: NSGA-II vs MOEA/D vs SPEA2', fontsize=14)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

# Determine overall best algorithm based on a combination of metrics
print("\nOVERALL CONCLUSION:")
if best_time_algo == best_priority_algo:
    print(f"The {best_time_algo} algorithm is the best overall solution as it provides both the minimum delivery time and maximum priority fulfillment.")
else:
    # Calculate a weighted score (lower is better)
    nsga2_score = min_time_solution_nsga2[0] / best_time_value - min_time_solution_nsga2[2] / (-best_priority_value)
    moead_score = min_time_solution_moead[0] / best_time_value - min_time_solution_moead[2] / (-best_priority_value)
    spea2_score = min_time_solution_spea2[0] / best_time_value - min_time_solution_spea2[2] / (-best_priority_value)
    
    scores = {
        'NSGA-II': nsga2_score, 
        'MOEA/D': moead_score, 
        'SPEA2': spea2_score
    }
    
    best_overall = min(scores, key=scores.get)
    print(f"The {best_overall} algorithm provides the best balance between delivery time and priority fulfillment.")
    print("However, the choice depends on your specific priorities:")
    print(f"- For minimum delivery time: Choose {best_time_algo}")
    print(f"- For maximum priority fulfillment: Choose {best_priority_algo}")

# Show the comparison plot
plt.tight_layout()
plt.show()

