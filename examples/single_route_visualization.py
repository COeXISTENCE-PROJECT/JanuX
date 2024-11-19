import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))
print(os.getcwd())

from visualizers import show_single_route

# File paths
nod_file_path = 'examples/network_files/csomor1.nod.xml'
edg_file_path = 'examples/network_files/csomor1.edg.xml'

# Paths
path = ['154551772#1', '115604051#0', '-115602933#5', '-115602933#4', '115604048#1', '279952229#3', \
    '279952229#4', '279952229#5', '-115604057#0', '115604057#0', '115604057#1', '-115604057#1']

# Origin and destination
origin, destination = '154551772#1', '-115604057#1'

xcrop = (1500, 3000)
ycrop = (300, 1200)

save_figure_to = 'examples/figures/single_path.png'

if __name__ == "__main__":
    show_single_route(nod_file_path, edg_file_path, path, origin, destination, xcrop=xcrop, ycrop=ycrop, save_file_path=save_figure_to)