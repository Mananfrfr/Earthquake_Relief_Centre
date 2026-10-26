"""Optional OpenStreetMap export of Jaipur major drivable roads."""
from pathlib import Path
import osmnx as ox


def main():
    output = Path(__file__).resolve().parents[1] / "data"
    output.mkdir(parents=True, exist_ok=True)
    graph_file = output / "jaipur_major_roads.graphml"
    image_file = output / "jaipur_major_roads.png"
    custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
    print("Downloading Jaipur major-road network from OpenStreetMap...")
    graph = ox.graph_from_place(
        "Jaipur, Rajasthan, India", network_type="drive", custom_filter=custom_filter
    )
    ox.save_graphml(graph, filepath=str(graph_file))
    ox.plot_graph(
        graph, node_size=0, edge_color="black", edge_linewidth=0.5,
        show=False, save=True, filepath=str(image_file), dpi=300, bgcolor="white"
    )
    print(f"Saved: {graph_file} and {image_file}")


if __name__ == "__main__":
    main()
