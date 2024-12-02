import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '././')))

import time

from graph_builders import build_digraph
from path_generators import calculate_free_flow_time
from path_generators import heuristic_generator
from utils import read_json
from visualizers import show_multi_routes
    
    
if __name__ == "__main__":
    
    ##################### PARAMS ############################

    network_name = "ingolstadt"
    connection_file_path = f"examples/network_files/{network_name}/{network_name}.con.xml"
    edge_file_path = f"examples/network_files/{network_name}/{network_name}.edg.xml"
    route_file_path = f"examples/network_files/{network_name}/{network_name}.rou.xml"
    nod_file_path = f"examples/network_files/{network_name}/{network_name}.nod.xml"

    ods = read_json(f"examples/network_files/{network_name}/ods.json")
    origins = ods["origins"]
    destinations = ods["destinations"]

    show_routes = True
    csv_save_path = "examples/results/"
    os.makedirs(csv_save_path, exist_ok=True)
    figures_save_path = "examples/figures/"
    os.makedirs(figures_save_path, exist_ok=True)

    path_gen_kwargs = {
        "random_seed": 42,      # For reproducibility
        "num_samples": 50,      # Number of samples to generate
        "number_of_paths": 3,   # Number of paths to find for each origin-destination pair
        "beta": -3,             # Beta parameter for the path generation
        "max_path_length": None,# Maximum length of the path, None for no limit
        "allow_loops": False,   # Allow loops in the path
        "adaptive": True,       # Use adaptive sampling (Shifts beta when it gets stuck)
    }

    visualization_kwargs = {
        "autocrop": True,       # Automatically crop the network boundaries
        "xcrop": (1500, 3000),  # x-axis crop boundaries (only used if autocrop is False)
        "ycrop": (300, 1200)
    }

    def heuristic1(path1, path2, path3, network):
        return -sum(len(path) for path in [path1, path2, path3])

    def heuristic2(path1, path2, path3, network):
        return -sum(calculate_free_flow_time(path, network) for path in [path1, path2, path3])

    def heuristic3(path1, path2, path3, network):
        common_edges1 = set(path1).intersection(set(path2))
        common_edges2 = set(path1).intersection(set(path3))
        common_edges3 = set(path2).intersection(set(path3))
        max_similarity = max(len(common_edges1), len(common_edges2), len(common_edges3))
        return -max_similarity

    heuristics = [heuristic1, heuristic2, heuristic3]
    heur_weights = [0.2, 0.5, 0.3]

    ########################################################

    start_time = time.time()
    # Generate network and paths
    network = build_digraph(connection_file_path, edge_file_path, route_file_path)
    routes = heuristic_generator(network, origins, destinations, heuristics, heur_weights, **path_gen_kwargs)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    if show_routes:
        # Visualize paths
        for origin_idx, origin in enumerate(origins):
            for dest_idx, destination in enumerate(destinations):
                routes_to_show = routes[(routes["origins"] == origin) & (routes["destinations"] == destination)]['path']
                routes_to_show = [route.split(",") for route in routes_to_show]# Specify the save path and title for the figure
                fig_save_path = os.path.join(figures_save_path, f"{network_name}_{origin_idx}_{dest_idx}.png")
                title=f"Origin: {origin_idx} ({origin}), Destination: {dest_idx} ({destination})"
                visualization_kwargs.update({"save_file_path": fig_save_path, "title": title})
                # Show the routes
                show_multi_routes(nod_file_path, edge_file_path, routes_to_show, origin, destination, **visualization_kwargs)
    
    csv_save_path = os.path.join(csv_save_path, f"{network_name}_routes.csv")
    routes.to_csv(csv_save_path, index=False)