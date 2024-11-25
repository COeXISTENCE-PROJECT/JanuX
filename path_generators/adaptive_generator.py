import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import logging
import networkx as nx
import pandas as pd

from path_generators import calculate_free_flow_time
from path_generators import paths_to_df
from path_generators.basic_generator import BasicPathGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

class AdaptivePathGenerator(BasicPathGenerator):
    
    """
    AdaptivePathGenerator extends the functionality of BasicPathGenerator to dynamically adjust
    sampling parameters (beta and max_path_length) when generating routes for origin-destination pairs.
    For more information on the BasicPathGenerator, refer to the class documentation.
    
    Attributes:
        network (nx.DiGraph): The directed graph representing the transportation network.
        origins (list[str]): A list of origin node IDs.
        destinations (list[str]): A list of destination node IDs.
        tolerance (int | None): Maximum number of iterations allowed before adjusting sampling parameters.
            Defaults to None, which is set to twice the number of samples.
        shift_params (int): Percentage by which beta and max_path_length are adjusted if the tolerance is exceeded.
            Defaults to 5%.
    """
    
    def __init__(self, 
                 network: nx.DiGraph, 
                 origins: list[str], 
                 destinations: list[str], 
                 tolerance: int | None = None,
                 shift_params : int = 5,
                 **kwargs):
        super().__init__(network, origins, destinations, **kwargs)
        self.tolerance = tolerance
        self.shift_params = shift_params

    def generate_routes(self, as_df: bool = True) -> pd.DataFrame | dict:
        
        """
        Generates paths for each origin-destination pair in the network, dynamically adjusting parameters
        (beta and max_path_length) if the sampling process exceeds the tolerance limit.

        Args:
            as_df (bool): Whether to return the results as a pandas DataFrame. Defaults to True.

        Returns:
            pd.DataFrame | dict: If as_df is True, returns a DataFrame containing the sampled routes,
            free-flow times, and related metadata. Otherwise, returns a dictionary where keys are 
            tuples of (origin_id, destination_id) and values are lists of paths.

        Raises:
            AssertionError: If the following conditions are not met:
                - num_samples >= number_of_paths
                - max_path_length > 0
                - shift_params > 0
                - beta < 0

        Notes:
            - Adjusts beta and max_path_length dynamically when the sampling process gets stuck.
            - Beta is increased (becomes less negative) by `shift_params`%, and max_path_length is increased
              by `shift_params`% if tolerance is exceeded.
            - Logs warnings and parameter adjustments during the adaptive process.
        """
        
        assert self.num_samples >= self.number_of_paths, f"Number of samples ({self.num_samples}) should be \
            at least equal to the number of routes ({self.number_of_paths})"
        assert self.max_path_length > 0, f"Maximum path length should be greater than 0"
        assert self.shift_params > 0, f"Shift parameter should be greater than 0"
        assert self.beta < 0, f"Beta should be less than 0"
        
        if self.tolerance is None:
            self.tolerance = self.num_samples * 2
            
        routes = dict()   # Tuple<od_id, dest_id> : List<routes>
        for dest_idx, dest_name in self.destinations.items():
            node_potentials = dict(nx.shortest_path_length(self.network, target=dest_name, weight=self.weight))
            for origin_idx, origin_name in self.origins.items():
                sampled_routes = list()   # num_samples number of routes
                iteration_count = 0
                initial_beta, initial_max_path_len = self.beta, self.max_path_length
                logging.info(f"Sampling paths for {origin_idx} -> {dest_idx} with beta: {self.beta}, max_path_length: {self.max_path_length}")
                while (len(sampled_routes) < self.num_samples) or (len(set(sampled_routes)) < self.number_of_paths):
                    # If this gets stuck, increase beta and max_path_length
                    if iteration_count > self.tolerance:
                        logging.warning(f"Exceeded tolerance for {origin_idx} -> {dest_idx}.")
                        shifted_beta = self.beta * ((100-self.shift_params) / 100) # Increase beta by shift_params%
                        self.beta = min(shifted_beta, -.01)
                        if self.max_path_length != float('inf'):
                            shifted_max_path_len = self.max_path_length * ((100+self.shift_params) / 100)
                            self.max_path_length = int(shifted_max_path_len) # Increase max_path_length by shift_params%
                        logging.info(f"Beta: {self.beta}, Max Path Length: {self.max_path_length}")
                        iteration_count = 0
                        
                    path = self._sample_single_route(origin_name, dest_name, node_potentials)
                    if not path is None:
                        sampled_routes.append(tuple(path))
                    iteration_count += 1
                    
                self.beta, self.max_path_length = initial_beta, initial_max_path_len
                logging.info(f"Sampled {len(sampled_routes)} paths for {origin_idx} -> {dest_idx}")
                routes[(origin_idx, dest_idx)] = self._pick_routes_from_samples(sampled_routes)
                logging.info(f"Selected {len(set(routes[(origin_idx, dest_idx)]))} paths for {origin_idx} -> {dest_idx}")
        if as_df:
            free_flows = {od: [calculate_free_flow_time(route, self.network) for route in routes[od]] for od in routes}
            routes_df = paths_to_df(routes, self.origins, self.destinations, free_flows)
            return routes_df
        else:
            return routes