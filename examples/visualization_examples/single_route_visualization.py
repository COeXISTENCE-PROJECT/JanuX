import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '././')))
print(os.getcwd())

from visualizers import show_single_route

"""
This script visualizes a single route in a transportation network and saves the visualization as an image.

Workflow:
1. The `show_single_route` function is called with the specified parameters.
2. The function visualizes the route defined by the `path` on the transportation network described in the provided node and edge files.
3. The visualization is cropped to the specified `xcrop` and `ycrop` ranges.
4. The resulting visualization is saved to the file specified in `save_figure_to`.

Output:
- A visualization of the route is saved as an image in the specified location.
"""

if __name__ == "__main__":
    
    ##################### PARAMS ############################

    network_name = "ingolstadt"

    # File paths
    nod_file_path = f"examples/network_files/{network_name}/{network_name}.nod.xml"
    edg_file_path = f"examples/network_files/{network_name}/{network_name}.edg.xml"

    # Paths
    path = "-24634511,24634510#1,24634510#6,204588664#0,-23436553#2,-23436553#1,-22690205#1,233675413#4,-25145011#2,-25145012#5,22716069#2,-653473569#1,-653473569#0,-653473568#1,-653473568#0,-18813598#8,-18813598#7,-18813598#6,-18813598#1,-201963522#9,-201963522#5,-201963522#4,-173177776#1"
    path = path.split(",")

    # Origin and destination
    origin, destination = path[0], path[-1]

    # Figure crop ranges for the visualization
    autocrop = True
    # or
    xcrop = (1500, 3000)
    ycrop = (300, 1200)

    save_figure_to = f'examples/figures/{network_name}_single_path.png'

    ########################################################
    
    show_single_route(nod_file_path, edg_file_path, path, origin, destination, autocrop=autocrop, xcrop=xcrop, ycrop=ycrop, save_file_path=save_figure_to)