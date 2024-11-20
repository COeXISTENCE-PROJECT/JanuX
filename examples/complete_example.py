import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

from graph_builders import build_digraph
from path_generators import basic_generator
from visualizers import show_multi_routes

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
    "number_of_paths": 3,
    "beta": -2
}

########################################################
    
if __name__ == "__main__":
    # Generate network and paths
    network = build_digraph(connection_file_path, edge_file_path, route_file_path)
    routes = basic_generator(network, origins, destinations, **kwargs)
    
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