import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import random
import time

from collections import Counter

from path_generators import adaptive_generator
from graph_builders import build_digraph
from utils import read_json
from visualizers import show_edge_attributes
from visualizers import show_multi_routes

"""
This script demonstrates generating paths in a transportation network and visualizing the results using predefined 
parameters for origins, destinations, and network files.

Workflow:
1. The directed graph (`network`) is constructed using `build_digraph` and the provided network files.
2. Paths are generated between all specified origin-destination pairs using the `basic_generator` function.
3. If `show_routes` is set to `True`:
   - For each origin-destination pair, the corresponding paths are visualized and saved as PNG files in the `examples/figures/` directory.
   - The visualization uses cropped network boundaries defined by `xcrop` and `ycrop`.
4. The generated routes are saved to a CSV file at the location specified by `routes_csv_path`.

Output:
- Visualizations of paths saved as PNG files in `examples/figures/`.
- A CSV file containing the generated paths with details such as origin, destination, path and free-flow travel time.
"""

##################### PARAMS ############################

network_name = "ingolstadt"
connection_file_path = f"examples/network_files/{network_name}/{network_name}.con.xml"
edge_file_path = f"examples/network_files/{network_name}/{network_name}.edg.xml"
route_file_path = f"examples/network_files/{network_name}/{network_name}.rou.xml"
nod_file_path = f"examples/network_files/{network_name}/{network_name}.nod.xml"

ods = read_json(f"examples/network_files/{network_name}/ods.json")
origins = ods["origins"]
destinations = ods["destinations"]

autocrop = True
xcrop = (1500, 3000)
ycrop = (300, 1200)

show_routes = True
routes_csv_path = "examples/results/routes.csv"

show_congestion = True
save_figure_to = 'examples/figures/congestion_visualization.png'

kwargs = {
    "random_seed": 42,
    "num_samples": 100,
    "number_of_paths": 3,
    "beta": -3.5,
    "max_path_length": None,
}

########################################################
    
if __name__ == "__main__":
    
    start_time = time.time()
    # Generate network and paths
    network = build_digraph(connection_file_path, edge_file_path, route_file_path)
    routes = adaptive_generator(network, origins, destinations, **kwargs)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    if show_routes:
        # Visualize paths
        for origin_idx, origin in enumerate(origins):
            for dest_idx, destination in enumerate(destinations):
                save_path = f"examples/figures/{origin_idx}_{dest_idx}.png"
                routes_to_show = routes[(routes["origins"] == origin) & (routes["destinations"] == destination)]['path']
                routes_to_show = [route.split(",") for route in routes_to_show]
                
                show_multi_routes(nod_file_path, edge_file_path, routes_to_show, origin, destination, autocrop=autocrop,\
                    xcrop=xcrop, ycrop=ycrop, save_file_path=save_path, title=f"Origin: {origin_idx}({origin}), Destination: {dest_idx}({destination})")
        
    routes.to_csv(routes_csv_path, index=False)
    
    
    if show_congestion:
        routes = routes["path"].values

        all_edges = list()
        for route in routes:
            all_edges += route.split(",")
            
        counts = Counter(all_edges)
        # normalize counts between 0-1
        max_count = max(counts.values())
        counts = {edge: count / max_count for edge, count in counts.items()}

        # Count based with some noise
        congestion_dict = {edge: min(counts[edge]+random.random()/5, 1.0) for edge in counts.keys()}
        
        show_edge_attributes(
            nod_file_path,
            edge_file_path,
            congestion_dict,
            save_file_path=save_figure_to,
        )
    
    