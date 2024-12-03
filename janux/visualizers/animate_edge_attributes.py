import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import logging

from concurrent.futures import ProcessPoolExecutor
from PIL import Image

from janux.visualizers.visualization_utils import create_graph
from janux.visualizers.visualization_utils import parse_network_files
from janux.visualizers.visualize_edge_attributes import visualize_congestion

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

#################################################

def animate_edge_attributes(nod_file_path: str,
                                  edg_file_path: str,
                                  congestion_dicts: list[dict],
                                  save_frames_dir: str,
                                  save_gif_to: str,
                                  frame_duration:int = 200,
                                  **kwargs):
    
    # assert all congestion dicts have the same shapes and keys
    assert all(congestion_dicts[0].keys() == congestion_dict.keys() for congestion_dict in congestion_dicts), "Congestion dictionaries must have the same keys, which are the edge IDs!"
    
    # Make sure path save_frames_path exists
    if not os.path.exists(save_frames_dir):
        os.makedirs(save_frames_dir)
    
    logging.info("Parsing graph data...")
    
    # Parse the network
    nodes, edges = parse_network_files(nod_file_path, edg_file_path)
    graph = create_graph(nodes, edges)
    
    logging.info("Generating frames...")
    
    num_frames = len(congestion_dicts)
    frame_paths = list()

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_frame, idx, congestion_dict, graph, save_frames_dir, num_frames
            )
            for idx, congestion_dict in enumerate(congestion_dicts)
        ]

        for future in futures:
            frame_paths.append(future.result())
               
    # Create GIF
    logging.info("Creating GIF...")
    frames = [Image.open(frame) for frame in frame_paths]
    frames[0].save(
        save_gif_to,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0
    )
    logging.info(f"Saved GIF to: {save_gif_to}")
    
    
def process_frame(idx, congestion_dict, graph, save_frames_dir, num_frames):
    save_figure_to = os.path.join(save_frames_dir, f'congestion_visualization_{idx}.png')
    visualize_congestion(
        graph,
        congestion_dict,
        show=False,
        save_file_path=save_figure_to,
        title=f"Congestion Visualization (Frame {idx+1}/{num_frames})"
    )
    return save_figure_to