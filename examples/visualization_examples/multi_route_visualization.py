import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '././')))

from janux import show_multi_routes

"""
This script visualizes multiple routes in a transportation network and saves the visualization as an image.
Workflow:
1. The `show_multi_routes` function is called with the specified parameters.
2. The function visualizes all the routes in `paths` on the transportation network described in the provided node and edge files.
3. The visualization is cropped to the specified `xcrop` and `ycrop` ranges.
4. The resulting visualization, showing all the routes on the same map, is saved to the file specified in `save_figure_to`.

Output:
- A visualization of the routes is saved as an image in the specified location.
"""

if __name__ == "__main__":
    
    ##################### PARAMS ############################

    network_name = "csomor"

    # File paths
    nod_file_path = f"examples/network_files/{network_name}/{network_name}.nod.xml"
    edg_file_path = f"examples/network_files/{network_name}/{network_name}.edg.xml"

    # Paths
    path1 = "441496282#0,-115604051#2,279952229#1,279952229#2,279952229#3,279952229#4,279952229#5,115604057#1,-115604057#1"
    path2 = "441496282#0,441496282#1,441496282#2,441496282#3,441496282#4,-115604047#2,279952229#5,115604057#1,-115604057#1"
    path3 = "441496282#0,441496282#1,-115604043#2,-115604043#1,-115602933#4,-115602933#3,-115602933#2,115604047#1,279952229#5,115604057#1,-115604057#1"

    paths = [path1, path2, path3]
    paths = [path.split(',') for path in paths] 

    # Origin and destination
    origin, destination = paths[0][0], paths[0][-1]

    # Figure crop ranges for the visualization
    autocrop = True
    # or
    xcrop = (1500, 3000)
    ycrop = (300, 1200)

    save_figs_path = f"examples/figures/"
    os.makedirs(save_figs_path, exist_ok=True)
    save_figure_to = f'{save_figs_path}{network_name}_multi_paths.png'

    ########################################################

    show_multi_routes(nod_file_path, edg_file_path, paths, origin, destination, autocrop=autocrop, xcrop=xcrop, ycrop=ycrop, save_file_path=save_figure_to)