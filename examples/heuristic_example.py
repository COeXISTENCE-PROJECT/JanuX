import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

from graph_builders import build_digraph
from path_generators import calculate_free_flow_time
from path_generators import heuristic_generator
from visualizers import show_multi_routes

"""
This script demonstrates generating paths in a transportation network using heuristic-based selection and visualizing the results.

Heuristicsv used in this example:
- `heuristic1`: Minimizes the total number of edges across all paths.
- `heuristic2`: Minimizes the total free-flow travel time across all paths.
- `heuristic3`: Minimizes the maximum edge overlap between paths to encourage diversity.

Workflow:
1. The network is constructed using `build_digraph` from the provided connection, edge, and route files.
2. Paths are generated using the `heuristic_generator` with the defined heuristics and weights.
3. If `show_routes` is enabled:
   - Each origin-destination pair's routes are visualized and saved as images in the `examples/figures/` directory.
4. The generated routes are saved to a CSV file in the specified `routes_csv_path`.

Output:
- Visualizations of routes saved as PNG files in `examples/figures/`.
- A CSV file containing the generated routes with details such as origin, destination, and path.
"""

##################### PARAMS ############################

origins = ["441496282#0", "154551772#1"]
destinations = ["-115604057#1", "-279952229#4"]

connection_file_path = "examples/network_files/csomor1.con.xml"
edge_file_path = "examples/network_files/csomor1.edg.xml"
route_file_path = "examples/network_files/csomor1.rou.xml"
nod_file_path = "examples/network_files/csomor1.nod.xml"

xcrop = (1500, 3000)
ycrop = (300, 1200)

show_routes = True
routes_csv_path = "examples/results/routes.csv"

kwargs = {
    "random_seed": 42,
    "num_samples": 100,
    "number_of_paths": 3
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
    
if __name__ == "__main__":
    # Generate network and paths
    network = build_digraph(connection_file_path, edge_file_path, route_file_path)
    routes = heuristic_generator(network, origins, destinations, heuristics, heur_weights, **kwargs)
    
    if show_routes:
        # Visualize paths
        for origin_idx, origin in enumerate(origins):
            for dest_idx, destination in enumerate(destinations):
                save_path = f"examples/figures/{origin_idx}_{dest_idx}.png"
                routes_to_show = routes[(routes["origins"] == origin) & (routes["destinations"] == destination)]['path']
                routes_to_show = [route.split(",") for route in routes_to_show]
                
                show_multi_routes(nod_file_path, edge_file_path, routes_to_show, origin, destination, \
                    xcrop=xcrop, ycrop=ycrop, save_file_path=save_path, title=f"Origin: {origin_idx}({origin}), Destination: {dest_idx}({destination})")
        
    routes.to_csv(routes_csv_path, index=False)