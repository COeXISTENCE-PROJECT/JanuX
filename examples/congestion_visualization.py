import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import pandas as pd
import random

from collections import Counter

from visualizers import show_edge_attributes

"""
This script visualizes congestion levels in a transportation network based on edge attributes.

Workflow:
1. The `show_edge_attributes` function is called with the specified parameters.
2. The function visualizes the congestion levels on the transportation network described in the provided node and edge files.
3. Congestion levels are represented by edge colors, with darker colors indicating higher congestion.
4. The resulting visualization is saved to the file specified in `save_figure_to`.

Output:
- A visualization of the congestion levels is saved as an image in the specified location.
"""

##################### PARAMS ############################

network_name = "ingolstadt"

# File paths
nod_file_path = f"examples/network_files/{network_name}/{network_name}.nod.xml"
edg_file_path = f"examples/network_files/{network_name}/{network_name}.edg.xml"

# read results/routes.csv
routes = pd.read_csv("examples/results/routes.csv")
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

save_figure_to = 'examples/figures/congestion_visualization.png'

########################################################

if __name__ == "__main__":
    show_edge_attributes(
        nod_file_path,
        edg_file_path,
        congestion_dict,
        save_file_path=save_figure_to
    )