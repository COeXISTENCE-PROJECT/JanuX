import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

from mock_keychain import Keychain as kc
from utils import get_params

from generate_digraph import generate_network
from generate_paths import generate_paths


if __name__ == "__main__":
    params = get_params("test/mock_params.json")
    gen_params = params[kc.PATH_GEN]

    # Get origins and destinations
    origins = gen_params[kc.ORIGINS]
    destinations = gen_params[kc.DESTINATIONS]

    # Get necessary files for network creation
    sim_params = params[kc.SIMULATOR]
    connection_file_path = "test/network_files/" + sim_params[kc.CONNECTION_FILE_PATH]
    edge_file_path = "test/network_files/" + sim_params[kc.EDGE_FILE_PATH]
    route_file_path = "test/network_files/" + sim_params[kc.ROUTE_FILE_PATH]

    # Generate network and paths
    network = generate_network(connection_file_path, edge_file_path, route_file_path)
    routes = generate_paths(network, origins, destinations, num_samples=100, number_of_paths=3)
    
    #df_to_prettytable(routes)
    routes.to_csv("test/routes.csv", index=False)