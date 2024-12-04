import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '././')))

import pandas as pd
import time

import janux as jx


"""
This script demonstrates the generation of paths between origins and destinations in a transportation network using
specific midpoints. It builds a directed graph from network files, generates paths through midpoints, and visualizes 
the resulting routes.

Workflow:
1. The script builds a directed graph using network files.
2. Paths are generated from the origin to each midpoint, and then from each midpoint to the destination.
3. The generated routes are concatenated and stored in a DataFrame.
4. If `show_routes` is enabled, the paths are visualized and saved as images.
5. The generated routes are saved to a CSV file.

Output:
- Visualizations of the generated routes are saved as PNG files in the `examples/figures/` directory.
- The list of routes is saved as a CSV file in the specified `csv_save_path`.
"""

if __name__ == "__main__":
    
    ##################### PARAMS ############################

    network_name = "csomor"
    connection_file_path = f"examples/network_files/{network_name}/{network_name}.con.xml"
    edge_file_path = f"examples/network_files/{network_name}/{network_name}.edg.xml"
    route_file_path = f"examples/network_files/{network_name}/{network_name}.rou.xml"
    nod_file_path = f"examples/network_files/{network_name}/{network_name}.nod.xml"

    origins = ["441496282#0"]
    destinations = ["-115604057#1"]

    #mid_points = ["-115602933#4", "441496282#4", "279952229#4"]
    mid_points = ["279952229#4", "-115604047#2", "115604047#1"]
    #mid_points = ["-115602933#0", "279952229#5", "-819269916#3"]

    show_routes = True
    csv_save_path = "examples/results/"
    os.makedirs(csv_save_path, exist_ok=True)
    figures_save_path = "examples/figures/"
    os.makedirs(figures_save_path, exist_ok=True)

    kwargs = {
        "random_seed": 42,
        "num_samples": 50,
        "number_of_paths": 1,
        "beta" : -2
    }

    visualization_kwargs = {
        "autocrop": True,       # Automatically crop the network boundaries
        "xcrop": (1500, 3000),  # x-axis crop boundaries (only used if autocrop is False)
        "ycrop": (300, 1200)
    }

    ########################################################
    
    start_time = time.time()
    # Generate network and paths
    network = jx.build_digraph(connection_file_path, edge_file_path, route_file_path)
    routes = list()
    for mid_point in mid_points:
        route_to_midpoint = jx.extended_generator(network, origins, [mid_point], as_df=True, calc_free_flow=True, **kwargs)
        route_to_midpoint = route_to_midpoint["path"].values[0]
        route_to_midpoint = jx.utils.iterable_to_string(route_to_midpoint.split(",")[:-1], ',')
        
        route_from_midpoint = jx.extended_generator(network, [mid_point], destinations, as_df=True, calc_free_flow=True, **kwargs)
        route_from_midpoint = route_from_midpoint["path"].values[0]
        
        routes.append({
            "origins": origins[0],
            "destinations": destinations[0],
            "path": route_to_midpoint + "," + route_from_midpoint
        })
        
    # Merge routes
    routes = pd.DataFrame(routes)
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
                jx.show_multi_routes(nod_file_path, edge_file_path, routes_to_show, origin, destination, **visualization_kwargs)
      
    csv_save_path = os.path.join(csv_save_path, f"{network_name}_routes.csv")  
    routes.to_csv(csv_save_path, index=False)