import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

from visualizers import show_multi_routes

# File paths
nod_file_path = 'examples/network_files/csomor1.nod.xml'
edg_file_path = 'examples/network_files/csomor1.edg.xml'

# Paths
path1 = ['154551772#1', '115604051#0', '-115602933#5', '-115602933#4', '115604048#1', '279952229#3', \
    '279952229#4', '279952229#5', '-115604057#0', '115604057#0', '115604057#1', '-115604057#1']
path2 = ['154551772#1', '154551772#2', '154551772#3', '115604043#0', '-115602933#4', '115604048#1', \
    '279952229#3', '279952229#4', '279952229#5', '115604057#1', '-115604057#1']
path3 = ['154551772#1', '115604051#0', '-115602933#5', '-115604043#0', '115604043#0', '-115602933#4',\
    '-115602933#3', '115604050#1', '279952229#4', '279952229#5', '115604057#1', '-115604057#1']
paths = [path1, path2, path3]

# Origin and destination
origin, destination = '154551772#1', '-115604057#1'

xcrop = (1500, 3000)
ycrop = (300, 1200)

save_figure_to = 'examples/figures/multi_paths.png'

if __name__ == "__main__":
    show_multi_routes(nod_file_path, edg_file_path, paths, origin, destination, xcrop=xcrop, ycrop=ycrop, save_file_path=save_figure_to)