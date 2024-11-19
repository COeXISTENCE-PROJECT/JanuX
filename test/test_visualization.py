import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

from visualize_paths import show_paths


if __name__ == "__main__":
    # File paths
    nod_file_path = 'test/network_files/csomor1.nod.xml'
    edg_file_path = 'test/network_files/csomor1.edg.xml'

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

    show_paths(nod_file_path, edg_file_path, paths, origin, destination, xcrop=(1500, 3000), ycrop=(0, 900), save_path='test/paths.png')