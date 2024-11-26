import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import pandas as pd
import random

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image

from visualizers import show_edge_attributes

##################### PARAMS ############################

# File paths
network_name = "ingolstadt"
nod_file_path = f"examples/network_files/{network_name}/{network_name}.nod.xml"
edg_file_path = f"examples/network_files/{network_name}/{network_name}.edg.xml"

num_frames = 50
frame_duration = 200

read_routes_from = "examples/results/routes.csv"
save_figure_path = "examples/figures/congestions/"
save_gif_to = "examples/figures/congestion_visualization.gif"

########################################################

if __name__ == "__main__":
    
    # Make sure path save_figure_path exists
    if not os.path.exists(save_figure_path):
        os.makedirs(save_figure_path)
    
    # mkae sure you have paths saved for Ingolstadt
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
        
    frame_paths = list()
    for idx, congestion_dict in enumerate(congestion_dicts):
        print(f"\rGenerating frame {idx+1}/{num_frames}", end="")
        save_figure_to = os.path.join(save_figure_path, f'congestion_visualization_{idx}.png')
        frame_paths.append(save_figure_to)
        show_edge_attributes(
            nod_file_path,
            edg_file_path,
            congestion_dict,
            save_file_path=save_figure_to,
            show=False,
            title=f"Congestion Visualization (Frame {idx+1}/{num_frames})"
        )
        plt.close()
    
    # Create GIF
    print("\nCreating GIF...")
    frames = [Image.open(frame) for frame in frame_paths]
    frames[0].save(
        save_gif_to,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0
    )
