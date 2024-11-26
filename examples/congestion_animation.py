import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import pandas as pd
import random

from collections import Counter

from visualizers import animate_edge_attributes

##################### PARAMS ############################

# File paths
network_name = "ingolstadt"
nod_file_path = f"examples/network_files/{network_name}/{network_name}.nod.xml"
edg_file_path = f"examples/network_files/{network_name}/{network_name}.edg.xml"

num_frames = 100
frame_duration = 200

read_routes_from = "examples/results/routes.csv"
save_frames_path = "examples/figures/congestions/"
save_gif_to = "examples/figures/congestion_animation.gif"

########################################################

if __name__ == "__main__":
    
    ##################### Create mock edge attributes ############################
    # !!! For this example, mkae sure you have paths saved for Ingolstadt !!!
    routes = pd.read_csv(read_routes_from)
    routes = routes["path"].values

    # Put axll edges in a single list
    all_edges = list()
    for route in routes:
        all_edges += route.split(",")
        
    # Count the number of times each edfge appears
    counts = Counter(all_edges)
    # Normalize counts between 0-.5
    max_count = max(counts.values())
    og_congestion_dict = {edge: (count / max_count) / 2 for edge, count in counts.items()}

    congestion_dicts = list()
    for i in range(num_frames):
        congestion_dict = {edge: min(og_congestion_dict[edge]+random.random()/4, 1.0) for edge in og_congestion_dict.keys()}
        congestion_dicts.append(congestion_dict)
        
    #################### Animate ############################ 
    
    animate_edge_attributes(nod_file_path,
                            edg_file_path,
                            congestion_dicts,
                            save_frames_dir=save_frames_path,
                            save_gif_to=save_gif_to,
                            frame_duration=frame_duration)
