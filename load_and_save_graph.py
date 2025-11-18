import osmnx as ox
import os

# === CONFIGURATION ===
place_name = "Jaipur, Rajasthan, India"  # Place name of the city
save_path = r"D:\OneDrive\Desktop\Coding\Python\BTP\Saved_graph\jaipur_drive6.graphml"


# Define main roads (good for routing)
#custom_filter = ('["highway"~"motorway|trunk|primary|secondary|tertiary"]')
custom_filter = ('["highway"~"motorway|trunk|primary|secondary|tertiary|"]')



# === LOAD JAIPUR DRIVABLE ROAD NETWORK USING PLACE NAME ===
print("⏳ Extracting Jaipur drivable road network from place name...")
G = ox.graph_from_place(place_name, network_type='drive', custom_filter=custom_filter)

# # Allowed highway types
# allowed_types = ["motorway", "trunk", "primary", "secondary", "tertiary"]
#
# # Filter edges
# edges_to_keep = [(u, v, k) for u, v, k, data in G.edges(keys=True, data=True)
#                  if isinstance(data.get("highway"), str) and data["highway"] in allowed_types
#                  or isinstance(data.get("highway"), list) and any(h in allowed_types for h in data["highway"])]
#
# # Create subgraph with only desired edges
# G_main_roads = G.edge_subgraph(edges_to_keep).copy()

# === SAVE GRAPH FOR FUTURE USE ===
os.makedirs(os.path.dirname(save_path), exist_ok=True)
ox.save_graphml(G, save_path)

print(f"✅ Jaipur road network saved to: {save_path}")



# Plot and save the entire graph directly (osmnx will handle the graph itself, not GeoDataFrames)
fig, ax = ox.plot_graph(
    G,
    node_size=0,
    edge_color="black",
    edge_linewidth=0.5,
    show=False,
    save=True,
    filepath="jaipur_roads6.png",
    dpi=300,
    bgcolor="white"
)
