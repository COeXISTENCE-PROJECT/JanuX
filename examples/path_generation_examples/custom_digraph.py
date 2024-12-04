import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '././')))

import janux as jx
import networkx as nx
import numpy as np

seed = 42
rng = np.random.default_rng(seed)

#################################################
# Generate random directed graph

num_nodes = 500
probability_conn = 0.01
G = nx.gnp_random_graph(num_nodes, probability_conn, directed=True, seed=seed)
# Relabel nodes to strings
G = nx.relabel_nodes(G, {node: str(node) for node in G.nodes})

print("Graph generated:", G)

#################################################
# Sample at random two origins and destinations

num_origins, num_destinations = 2, 2
ods_connected = False
while not ods_connected:
    ods = rng.choice(G.nodes, num_origins+num_destinations, replace=False)
    origins, destinations = ods[:num_origins], ods[num_origins:]
    # Check if there is a path between the selected origins and destinations
    ods_connected = all(nx.has_path(G, o, d) for o in origins for d in destinations)
    
print("Selected origins:", origins)
print("Selected destinations:", destinations)

#################################################
# Generation with basic generator

# Generate routes
num_samples, num_paths, beta = 20, 3, -3
routes = jx.basic_generator(G, origins, destinations, as_df=False,
                               num_samples=num_samples, number_of_paths=num_paths,
                               beta=beta, random_seed=seed,
                               verbose=False)

#################################################
# Print routes

print("\nRoutes generated with Basic Generator:")
for od in routes:
    print(f"Origin: {od[0]}, Destination: {od[1]}")
    for idx, route in enumerate(routes[od]):
        print(f"Route {idx+1}: {route}")
        
#################################################

