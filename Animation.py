import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import pygame
import sys
import math

ROUTE_WEIGHT = "length"   # or "length"







# === Load saved road network graph ===
graph_path = r"D:\OneDrive\Desktop\Coding\Python\BTP\Saved_graph\jaipur_drive2.graphml"
print("⏳ Loading saved Jaipur graph...")
G = ox.load_graphml(graph_path)
print("✅ Graph loaded!")

# Now you can proceed with:
# - Plotting the graph
# - Finding nearest nodes to RCs and DSs
# - Calculating routes
# - Running NSGA-II
# - Animating with Pygame




# def print_graph_summary(G):
#     print("✅ Graph Summary:")
#     print(f"• Nodes: {len(G.nodes)}")
#     print(f"• Edges: {len(G.edges)}")
#     print(f"• Is directed: {nx.is_directed(G)}")
#     print(f"• Is strongly connected: {nx.is_strongly_connected(G) if nx.is_directed(G) else 'N/A'}")
#
#
# print_graph_summary(G)
#
#
#
# # Visual check of the loaded graph
# fig, ax = ox.plot_graph(G, node_size=5, edge_color="gray", edge_linewidth=0.5, bgcolor="white")
#
# plt.show()


relief_centers = [
    {"name": "RC1", "lat": 26.84610, "lon": 75.80479},
    {"name": "RC2", "lat": 26.91058, "lon": 75.82432},
    {"name": "RC3", "lat": 26.83613, "lon": 75.87966},
    {"name": "RC4", "lat": 26.91395, "lon": 75.73120},
    {"name": "RC5", "lat": 26.94282, "lon": 75.80298},
]

# # Define relief centers and disaster sites
# relief_centers = [
#     {"name": "RC1", "lat": 26.95188, "lon": 75.80537},
#     {"name": "RC2", "lat": 26.99605, "lon": 75.86116},
#     {"name": "RC3", "lat": 26.84615, "lon": 75.80592},
#     {"name": "RC4", "lat": 26.91381, "lon": 75.81876},
#     {"name": "RC5", "lat": 26.91141, "lon": 75.73084},
# ]

# RC 1: Lat = 26.89817, Lon = 75.73853
# RC 2: Lat = 26.84606, Lon = 75.80521
# RC 3: Lat = 26.99900, Lon = 75.85989
# RC 4: Lat = 26.95154, Lon = 75.80543
# RC 5: Lat = 26.89233, Lon = 75.82742





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



# Get nearest nodes
rc_nodes = ox.distance.nearest_nodes(G, [rc["lon"] for rc in relief_centers], [rc["lat"] for rc in relief_centers])
ds_nodes = ox.distance.nearest_nodes(G, [ds["lon"] for ds in disaster_sites], [ds["lat"] for ds in disaster_sites])

# Create mappings
rc_node_mapping = {relief_centers[i]["name"]: int(rc_nodes[i]) for i in range(len(rc_nodes))}
ds_node_mapping = {disaster_sites[i]["name"]: {"node": int(ds_nodes[i]), "priority": disaster_sites[i]["priority"]} for i in range(len(ds_nodes))}



# Convert graph nodes to GeoDataFrame
nodes, _ = ox.graph_to_gdfs(G)

# Function to get lat/lon from node ID
def get_lat_lon(node_id):
    row = nodes.loc[node_id]
    return (float(row.y), float(row.x))  # (lat, lon)

# Update Relief Center Mapping with coordinates
for rc_name, node_id in rc_node_mapping.items():
    latlon = get_lat_lon(node_id)
    rc_node_mapping[rc_name] = {
        "node": node_id,
        "coords": latlon
    }

# Update Disaster Site Mapping with coordinates
for ds_name, ds_info in ds_node_mapping.items():
    node_id = ds_info["node"]
    latlon = get_lat_lon(node_id)
    ds_info["coords"] = latlon


print("Relief Center Node Mapping:")
print(rc_node_mapping)
print("\nDisaster Site Node Mapping:")
print(ds_node_mapping)



# # Initialize Pygame
# pygame.init()
#
# # Set up the Pygame window
# screen = pygame.display.set_mode((800, 600))
# pygame.display.set_caption("Disaster Management Simulation")
#
# # Load the background image (road network map)
# background = pygame.image.load('jaipur_roads.png')
#
# # Main loop to display the image
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#
#     # Draw the background image
#     screen.blit(background, (0, 0))
#
#     # Update display
#     pygame.display.update()
#
#     # Resize the image to fit the screen
#     background = pygame.transform.scale(background, (800, 600))
#
# pygame.quit()

# Step 2.1: Compute shortest paths between RCs and DSs (by distance)
print("\n📍 Computing shortest paths between Relief Centers and Disaster Sites...\n")

distance_paths = {}

for rc_name,rc_info in rc_node_mapping.items():
    rc_node = rc_info["node"]
    for ds_name,ds_info in ds_node_mapping.items():
        ds_node = ds_info["node"]
        try:
            path = nx.shortest_path(G, rc_node, ds_node, weight="length")
            length = nx.shortest_path_length(G, rc_node, ds_node, weight="length")
            distance_paths[(rc_name, ds_name)] = {
                "path": path,
                "length_meters": length
            }
            print(f"✅ Path from {rc_name} to {ds_name}: {length:.2f} meters")
        except nx.NetworkXNoPath:
            print(f"❌ No path between {rc_name} and {ds_name}")




# # Get the list of all node IDs
# node_ids = list(G.nodes())
#
# # Create a color list with default gray color
# node_colors = ['gray'] * len(node_ids)
#
# # Get the indices of the RC1 and Jagatpura nodes
# rc1_node = rc_node_mapping["RC1"]
# Jagatpura_node = ds_node_mapping["Jagatpura"]["node"]
#
# # If RC1 and Jagatpura nodes are in the list, change their colors
# if rc1_node in node_ids:
#     idx_rc1 = node_ids.index(rc1_node)
#     node_colors[idx_rc1] = 'green'
#
# if Jagatpura_node in node_ids:
#     idx_Jagatpura = node_ids.index(Jagatpura_node)
#     node_colors[idx_Jagatpura] = 'red'
#
# # Plot the graph with highlighted nodes
# fig, ax = ox.plot_graph(G,
#                         node_size=5,
#                         node_color=node_colors,
#                         edge_color="lightgray",
#                         bgcolor="white")

# # Step 1: Node IDs
# rc1_node = rc_node_mapping["RC1"]
# Jagatpura_node = ds_node_mapping["Jagatpura"]["node"]
#
# # Step 2: Get weakly connected components (for directed graph) or connected components (if undirected)
# if nx.is_directed(G):
#     components = list(nx.weakly_connected_components(G))
# else:
#     components = list(nx.connected_components(G))
#
# # Step 3: Find component for RC1 and Jagatpura
# rc1_component = next(c for c in components if rc1_node in c)
# Jagatpura_component = next(c for c in components if Jagatpura_node in c)
#
# # Step 4: Extract subgraphs
# rc1_subgraph = G.subgraph(rc1_component).copy()
# Jagatpura_subgraph = G.subgraph(Jagatpura_component).copy()
#
# # Step 5: Plot RC1 subgraph
# fig1, ax1 = ox.plot_graph(rc1_subgraph,
#                           node_size=[20 if n==rc1_node else 5 for n in rc1_subgraph.nodes()],
#                           node_color=['green' if n == rc1_node else 'gray' for n in rc1_subgraph.nodes()],
#                           edge_color='black',
#                           bgcolor='white')
#
# # Step 6: Plot Jagatpura subgraph
# fig2, ax2 = ox.plot_graph(Jagatpura_subgraph,
#                           node_size=[20 if n == Jagatpura_node else 5 for n in Jagatpura_subgraph.nodes()],
#                           node_color=['red' if n == Jagatpura_node else 'gray' for n in Jagatpura_subgraph.nodes()],
#                           edge_color='black',
#                           bgcolor='white')


# Define speeds in km/h
speed_map = {
    "motorway": 80,
    "trunk": 70,
    "primary": 60,
    "secondary": 50,
    "tertiary": 40,
    "residential": 30,
    "service": 20,
}

# Default speed if unknown
default_speed_kph = 25


# Function to get speed in m/s
def get_speed(edge):
    highway = edge.get("highway", None)

    # Handle if highway is a list
    if isinstance(highway, list):
        highway = highway[0]

    speed_kph = speed_map.get(highway, default_speed_kph)
    speed_mps = speed_kph * 1000 / 3600
    return speed_mps


# Assign travel time to each edge
for u, v, k, data in G.edges(keys=True, data=True):
    length = data.get("length", 1)  # fallback if missing
    speed = get_speed(data)
    travel_time_sec = length / speed
    data["travel_time"] = travel_time_sec



# # Print a few sample edges and their travel_time
# count = 0
# for u, v, data in G.edges(data=True):
#     if "travel_time" in data:
#         print(f"Edge from {u} to {v}: travel_time = {data['travel_time']:.2f} seconds")
#         count += 1
#     if count >= 5:
#         break


# Step 2.3: Compute shortest paths between RCs and DSs based on travel time

print("\n📍 Computing shortest travel time paths between Relief Centers and Disaster Sites...\n")

travel_time_paths = {}
travel_time_matrix = {}

for rc_name, rc_info in rc_node_mapping.items():
    rc_node = rc_info["node"]
    travel_time_matrix[rc_name] = {}
    for ds_name, ds_info in ds_node_mapping.items():
        ds_node = ds_info["node"]
        try:
            # Find the shortest path by travel time
            path = nx.shortest_path(G, rc_node, ds_node, weight="travel_time")
            travel_time = nx.shortest_path_length(G, rc_node, ds_node, weight="travel_time")

            travel_time_paths[(rc_name, ds_name)] = {
                "path": path,
                "travel_time_seconds": travel_time  # in seconds
            }

            # Store travel time and distance in matrices
            travel_time_matrix[rc_name][ds_name] = travel_time
            print(f"✅ Path from {rc_name} to {ds_name}: {travel_time:.2f} seconds")
        except nx.NetworkXNoPath:
            print(f"❌ No path between {rc_name} and {ds_name}")
            travel_time_matrix[rc_name][ds_name] = None

# Print out travel time matrix for validation
print("\nTravel Time Matrix (in seconds):")
for rc_name, ds_info in travel_time_matrix.items():
    print(f"{rc_name}: {ds_info}")




# Initialize dictionary to store path data
route_data = {}

# Iterate over each RC and DS pair
for rc_name, rc_node in rc_node_mapping.items():
    route_data[rc_name] = {}
    for ds_name, ds_info in ds_node_mapping.items():
        ds_node = ds_info["node"]
        try:
            # Get the shortest distance path and shortest time path
            distance_path = distance_paths.get((rc_name, ds_name), {}).get("path", [])
            time_path = travel_time_paths.get((rc_name, ds_name), {}).get("path", [])

            # Compute distance of shortest distance path
            distance_length = sum(G[u][v][0]["length"] for u, v in zip(distance_path[:-1], distance_path[1:]))

            # Compute travel time on the shortest distance path
            travel_time_on_distance_path = sum(
                G[u][v][0]["travel_time"] for u, v in zip(distance_path[:-1], distance_path[1:]))

            # Compute travel time of shortest time path
            travel_time = sum(G[u][v][0]["travel_time"] for u, v in zip(time_path[:-1], time_path[1:]))

            # Compute distance of the shortest time path
            distance_on_time_path = sum(G[u][v][0]["length"] for u, v in zip(time_path[:-1], time_path[1:]))

            # Store the data in the dictionary
            route_data[rc_name][ds_name] = {
                "path_nodes_distance": distance_path,
                "path_length": distance_length,
                "travel_time_on_distance_path": travel_time_on_distance_path,

                "path_nodes_time": time_path,
                "travel_time_seconds": travel_time,
                "path_length_on_time_path": distance_on_time_path
            }

            print(f"✅ Stored route {rc_name} → {ds_name} | Dist: {distance_length:.2f} m, "
                  f"Time (dist path): {travel_time_on_distance_path:.2f} s, "
                  f"Time (time path): {travel_time:.2f} s, Dist (time path): {distance_on_time_path:.2f} m")

        except nx.NetworkXNoPath:
            print(f"❌ No path between {rc_name} and {ds_name}")
            route_data[rc_name][ds_name] = None

# Print a few samples for validation
print("\n📦 Sample Route Data:")
for rc_name, ds_data in route_data.items():
    for ds_name, data in ds_data.items():
        if data is not None:
            print(f"From {rc_name} to {ds_name}:")
            print(f"  - 📏 Distance Path: {data['path_length']} m")
            print(f"  - ⏱️  Time on Distance Path: {data['travel_time_on_distance_path']} s")
            print(f"  - 🛣️  Time Path: {data['travel_time_seconds']} s")
            print(f"  - 📏 Distance on Time Path: {data['path_length_on_time_path']} m")
            print()


# NSGA-II Solution Matrix
nsga_solution = {
    'RC1': {'Purani Basti': 19, 'Sanganer': 57, 'Shastri Nagar': 29, 'Jagatpura': 22, 'Jawahar Nagar': 13},
    'RC2': {'Purani Basti': 51, 'Sanganer': 43, 'Jagatpura': 19, 'Jawahar Nagar': 12},
    'RC3': {'Vaishali Nagar': 41, 'Shastri Nagar': 20, 'Jagatpura': 49},
    'RC4': {'Chandpole': 41, 'Shastri Nagar': 39},
    'RC5': {'Chandpole': 39, 'Vaishali Nagar': 19, 'Shastri Nagar': 2, 'Jawahar Nagar': 5}
}







# ----------------------------------------
# CONFIGURATION
# ----------------------------------------


edges = ox.graph_to_gdfs(G, nodes=False)
# Window settings
WIDTH, HEIGHT = 1200, 800
BACKGROUND_COLOR = (245, 245, 245)
RC_COLOR = (0, 102, 204)
DS_COLOR = (204, 0, 0)
FONT_COLOR = (0, 0, 0)

# Map bounds
MAP_MIN_LAT, MAP_MAX_LAT = 26.8, 27.05
MAP_MIN_LON, MAP_MAX_LON = 75.7, 75.95

# Zoom settings
zoom = 1.0
ZOOM_FACTOR = 1.1
MIN_ZOOM, MAX_ZOOM = 0.5, 5.0
pan_offset = [0, 0]
dragging = False
drag_start = (0, 0)

# Initialize pygame
pygame.init()
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Disaster Management Visualization - Zoom & Panning")
font = pygame.font.SysFont("Arial", 16)

# Load Jaipur city map background
#jaipur_map = pygame.image.load("jaipur_roads2.png").convert()

# Geo to screen
# def geo_to_screen(lat, lon):
#     x_norm = (lon - MAP_MIN_LON) / (MAP_MAX_LON - MAP_MIN_LON)
#     y_norm = (MAP_MAX_LAT - lat) / (MAP_MAX_LAT - MAP_MIN_LAT)
#     x = int(x_norm * WIDTH * zoom + pan_offset[0])
#     y = int(y_norm * HEIGHT * zoom + pan_offset[1])
#     return x, y

def geo_to_screen(lat, lon):
    # Normalize coordinates
    x_norm = (lon - MAP_MIN_LON) / (MAP_MAX_LON - MAP_MIN_LON)
    y_norm = (MAP_MAX_LAT - lat) / (MAP_MAX_LAT - MAP_MIN_LAT)  # Flip y to match screen coords

    # Maintain zoom centered with proper scaling
    x = int((x_norm * WIDTH) * zoom + pan_offset[0])
    y = int((y_norm * HEIGHT) * zoom + pan_offset[1])
    return x, y


def draw_roads():
    road_color = (200, 200, 200)  # Light gray
    for _, row in edges.iterrows():
        if row['geometry'].geom_type == 'LineString':
            coords = list(row['geometry'].coords)
            points = [geo_to_screen(lat, lon) for lon, lat in coords]
            pygame.draw.lines(window, road_color, False, points, 1)

        elif row['geometry'].geom_type == 'MultiLineString':
            for linestring in row['geometry']:
                coords = list(linestring.coords)
                points = [geo_to_screen(lat, lon) for lon, lat in coords]
                pygame.draw.lines(window, road_color, False, points, 1)




# Draw RCs and DSs
def draw_nodes():
    for rc_name, rc_info in rc_node_mapping.items():
        x, y = geo_to_screen(*rc_info["coords"])
        pygame.draw.circle(window, RC_COLOR, (x, y), max(3, int(8 * zoom)))
        window.blit(font.render(rc_name, True, FONT_COLOR), (x + 10, y))

    for ds_name, ds_info in ds_node_mapping.items():
        x, y = geo_to_screen(*ds_info["coords"])
        pygame.draw.circle(window, DS_COLOR, (x, y), max(2, int(6 * zoom)))
        window.blit(font.render(ds_name, True, FONT_COLOR), (x + 10, y))




def draw_routes():
    for rc_name, assignments in nsga_solution.items():
        rc_node = rc_node_mapping[rc_name]["node"]
        for ds_name, value in assignments.items():
            if value > 0:
                ds_node = ds_node_mapping[ds_name]["node"]
                try:
                    path = nx.shortest_path(G, source=rc_node, target=ds_node, weight=ROUTE_WEIGHT)
                    path_coords = [geo_to_screen(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
                    #print(f"{rc_name} ➝ {ds_name} | Path nodes: {len(path)}")

                    pygame.draw.lines(window, (0,200,0), False, path_coords, 2)
                except nx.NetworkXNoPath:
                    print(f"No path from {rc_name} to {ds_name}")




class Vehicle:
    def __init__(self, path, resource_amount=100, total_capacity=100, color=(255, 165, 0), speed=0.005):
        self.path = path
        self.index = 0
        self.progress = 0.0
        self.speed = speed
        self.resource_amount = resource_amount
        self.total_capacity = total_capacity
        self.color = color

    def update(self):
        if self.index < len(self.path) - 1:
            self.progress += self.speed
            if self.progress >= 1.0:
                self.progress = 0.0
                self.index += 1

    def draw(self, surface, font):
        if self.index < len(self.path) - 1:
            x1, y1 = self.path[self.index]
            x2, y2 = self.path[self.index + 1]
            x = x1 + (x2 - x1) * self.progress
            y = y1 + (y2 - y1) * self.progress

            # Draw vehicle
            pygame.draw.circle(surface, self.color, (int(x), int(y)), 6)

            # Draw capacity bar
            bar_width = 30
            bar_height = 5
            fill_width = int(bar_width * self.resource_amount / self.total_capacity)
            bar_x = int(x - bar_width // 2)
            bar_y = int(y - 15)

            pygame.draw.rect(surface, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))  # background
            pygame.draw.rect(surface, (0, 200, 0), (bar_x, bar_y, fill_width, bar_height))  # fill

            # Draw amount text
            text_surface = font.render(str(self.resource_amount), True, (0, 0, 0))
            surface.blit(text_surface, (x - 10, y - 30))

def get_path_travel_time(path_nodes):
    time = 0
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        # If the graph is a MultiDiGraph, pick the first available edge
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            edge = list(edge_data.values())[0]
            time += edge.get("travel_time", 1)
    return time


vehicle_capacity = 30
vehicles = []

for rc_name, assignments in nsga_solution.items():
    rc_node = rc_node_mapping[rc_name]["node"]
    for ds_name, resource_amount in assignments.items():
        if resource_amount > 0:
            ds_node = ds_node_mapping[ds_name]["node"]
            try:
                path = nx.shortest_path(G, source=rc_node, target=ds_node, weight=ROUTE_WEIGHT)
                screen_path = [geo_to_screen(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]

                # Calculate how many vehicles needed
                full_vehicles = resource_amount // vehicle_capacity
                remaining = resource_amount % vehicle_capacity

                travel_time = get_path_travel_time(path)

                # Add full vehicles
                for _ in range(full_vehicles):
                    vehicles.append(Vehicle(
                        path=screen_path,
                        resource_amount=vehicle_capacity,
                        total_capacity=vehicle_capacity,
                        color=(255, 165, 0),
                        speed=80/travel_time
                    ))

                # Add final partial vehicle if needed
                if remaining > 0:
                    vehicles.append(Vehicle(
                        path=screen_path,
                        resource_amount=remaining,
                        total_capacity=vehicle_capacity,
                        color=(255, 165, 0),
                        speed=80/travel_time
                    ))

            except nx.NetworkXNoPath:
                print(f"No path from {rc_name} to {ds_name}")





# Main loop
running = True
while running:
    window.fill(BACKGROUND_COLOR)

    # Draw map background
    # scaled_map = pygame.transform.smoothscale(jaipur_map, (int(WIDTH * zoom), int(HEIGHT * zoom)))
    # window.blit(scaled_map, pan_offset)

    draw_roads()
    draw_nodes()
    draw_routes()  # Resource delivery routes based on NSGA-II + precomputed paths

    for vehicle in vehicles:
        vehicle.update()
        vehicle.draw(window, font)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Mouse wheel zoom
        elif event.type == pygame.MOUSEWHEEL:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            before_zoom = ((mouse_x - pan_offset[0]) / (WIDTH * zoom),
                           (mouse_y - pan_offset[1]) / (HEIGHT * zoom))

            if event.y > 0:
                zoom *= ZOOM_FACTOR
            elif event.y < 0:
                zoom /= ZOOM_FACTOR

            # Clamp zoom
            zoom = max(MIN_ZOOM, min(zoom, MAX_ZOOM))

            after_zoom = ((mouse_x - pan_offset[0]) / (WIDTH * zoom),
                          (mouse_y - pan_offset[1]) / (HEIGHT * zoom))

            pan_offset[0] += int((after_zoom[0] - before_zoom[0]) * WIDTH * zoom)
            pan_offset[1] += int((after_zoom[1] - before_zoom[1]) * HEIGHT * zoom)

        # Start dragging
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                dragging = True
                drag_start = pygame.mouse.get_pos()

        # Stop dragging
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging = False

        # Perform drag
        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                mx, my = pygame.mouse.get_pos()
                dx = mx - drag_start[0]
                dy = my - drag_start[1]
                pan_offset[0] += dx
                pan_offset[1] += dy
                drag_start = (mx, my)

    pygame.display.update()

pygame.quit()
sys.exit()