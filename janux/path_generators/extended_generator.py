import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import networkx as nx
import pandas as pd

from typing import List, Union

from janux.path_generators import calculate_free_flow_time
from janux.path_generators import paths_to_df
from janux.path_generators.basic_generator import BasicPathGenerator
from janux.utils import get_params

class ExtendedPathGenerator(BasicPathGenerator):
    
    """
    Extends the functionality of `BasicPathGenerator` to allow for adaptive parameter adjustments 
    and additional route constraints during path generation.

    This class provides enhanced features for generating routes in a transportation network, 
    including adaptive handling of situations where the desired number of unique paths cannot 
    be sampled with the current parameters. Parameters such as beta and maximum path length 
    can be adjusted dynamically to improve the success rate of route generation.

    Args:
        network (nx.DiGraph): The transportation network represented as a directed graph.
        origins (list[str]): A list of origin nodes in the network.
        destinations (list[str]): A list of destination nodes in the network.
        **kwargs: Additional parameters for customization. These parameters can override 
                  defaults from the `path_gen_params.json` file.

    Attributes:
        allow_loops (bool): Whether loops are allowed in the generated paths.
        adaptive (bool): Whether to adapt parameters dynamically when path sampling fails.
        tolerate_num_iterations (int): Number of failed iterations before triggering parameter adjustment.
        shift_parameters_by (float): Percentage by which parameters are shifted during adaptation.
        params_to_shift (str): Parameters to shift during adaptation. Options: "beta", "max_path_length", "both", "none".
        max_path_length (int | float): Maximum allowed length for any generated path. Defaults to infinity if not set.
    """
    
    def __init__(self, 
                 network: nx.DiGraph, 
                 origins: list[str], 
                 destinations: list[str], 
                 **kwargs):
        
        super().__init__(network, origins, destinations, **kwargs)
        
        # Determine the absolute path to the `path_gen_params.json` file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        params_file_path = os.path.join(current_dir, "path_gen_params.json")
        
        # Get parameters from the params.json file and update them with the provided kwargs
        params = get_params(params_file_path)
        params.update(kwargs)
        
        # Get parameters
        self.allow_loops = params["allow_loops"]
        
        self.adaptive = params["adaptive"]
        self.tolerate_num_iterations = params["tolerate_num_iterations"]
        self.shift_parameters_by = params["shift_parameters_by"]
        self.params_to_shift = params["params_to_shift"]
        
        self.max_path_length = params["max_path_length"]
        if self.max_path_length is None:
            self.max_path_length = float("inf")
        
        
    def generate_routes(self, as_df: bool = True, calc_free_flow: bool = False) -> Union[pd.DataFrame, dict]:
        
        """
        Generates routes between origin-destination pairs in the network.

        This method iteratively samples routes for each origin-destination pair and selects 
        unique paths based on the specified number of paths (`number_of_paths`) and sampling 
        parameters. If adaptive behavior is enabled, the method adjusts parameters like beta 
        and maximum path length when the desired number of unique paths cannot be achieved 
        within a set number of iterations.

        Args:
            as_df (bool): If True, returns the routes as a pandas DataFrame. If False, 
                          returns the routes as a dictionary. Defaults to True.
            calc_free_flow (bool): If True, calculates the free-flow time for each route.

        Returns:
            pd.DataFrame | dict: 
                - A DataFrame containing the routes, free-flow times, and related metadata if `as_df` is True.
                - A dictionary mapping (origin_id, destination_id) to lists of selected routes 
                  if `as_df` is False.

        Raises:
            AssertionError: If the number of samples (`num_samples`) is less than the number 
                            of required unique paths (`number_of_paths`).
            AssertionError: If `max_path_length` is not greater than 0.
            AssertionError: If `beta` is not less than 0.
            AssertionError: If `shift_parameters_by` is not greater than 0.
            AssertionError: If `params_to_shift` is not one of ["beta", "max_path_length", "both", "none"].

        Notes:
            - This method uses a logit-based probabilistic model to sample routes.
            - Adaptive adjustments to `beta` and `max_path_length` ensure the desired number 
              of unique paths can be generated.
            - Sampling and selection are logged for debugging and traceability.
        """
        
        assert self.num_samples >= self.number_of_paths, f"Number of samples ({self.num_samples}) should be \
            at least equal to the number of routes ({self.number_of_paths})"
        assert self.max_path_length > 0, f"Maximum path length should be greater than 0"
        assert self.beta < 0, f"Beta should be less than 0"
        assert self.shift_parameters_by > 0, f"Shift parameters should be greater than 0"
        assert self.params_to_shift in ["beta", "max_path_length", "both", "none"], f"Invalid parameter to shift: {self.params_to_shift}. Choose from 'beta', 'max_path_length', 'both', 'none'."
        
        routes = dict()   # Tuple<od_id, dest_id> : List<routes>
        for dest_idx, dest_name in self.destinations.items():
            node_potentials = dict(nx.shortest_path_length(self.network, target=dest_name, weight=self.weight))
            for origin_idx, origin_name in self.origins.items():
                sampled_routes = list()   # num_samples number of routes
                iteration_count = 0
                initial_beta, initial_max_path_len = self.beta, self.max_path_length
                while (len(sampled_routes) < self.num_samples) or (len(set(sampled_routes)) < self.number_of_paths):
                    
                    # If this gets stuck, increase beta and max_path_length
                    if (self.adaptive) and (iteration_count > self.tolerate_num_iterations):
                        self.logger.warning(f"Exceeded tolerance for {origin_idx} -> {dest_idx}.")
                        self.beta, self.max_path_length = self._shift_parameters(self.beta, self.max_path_length)
                        self.logger.info(f"Beta: {self.beta}, Max Path Length: {self.max_path_length}")
                        iteration_count = 0
                        
                    path = self._sample_single_route(origin_name, dest_name, node_potentials)
                    if not path is None:
                        sampled_routes.append(tuple(path))
                    iteration_count += 1
                
                self.beta, self.max_path_length = initial_beta, initial_max_path_len  
                self.logger.info(f"Sampled {len(sampled_routes)} paths for {origin_idx} -> {dest_idx}")
                routes[(origin_idx, dest_idx)] = self._pick_routes_from_samples(sampled_routes)
                self.logger.info(f"Selected {len(set(routes[(origin_idx, dest_idx)]))} paths for {origin_idx} -> {dest_idx}")
        if as_df:
            free_flows = None
            if calc_free_flow:
                free_flows = {od: [calculate_free_flow_time(route, self.network) for route in routes[od]] for od in routes}
            routes_df = paths_to_df(routes, self.origins, self.destinations, free_flows)
            return routes_df
        else:
            return routes


    def _sample_single_route(self, origin: str, destination: str, node_potentials: dict) -> Union[List[str], None]:
    
        """
        Samples a single route between an origin and a destination in the network.

        This method constructs a route incrementally by selecting the next node based on 
        a probabilistic logit model. The process stops when the destination is reached, 
        no valid neighbors are available, or the maximum path length is exceeded.

        Args:
            origin (str): The starting node of the route.
            destination (str): The target node of the route.
            node_potentials (dict): A dictionary mapping nodes to their potentials, 
                                    which influence the likelihood of selecting a node.

        Returns:
            list[str] | None: 
                - A list of nodes representing the sampled path if a valid route 
                  is successfully generated.
                - None if no valid route can be found under the given constraints.

        Notes:
            - If `allow_loops` is True, nodes can be revisited in the route. Otherwise, 
              visited nodes are excluded from future options.
            - If the destination is a direct neighbor of the current node, the route 
              is completed immediately.
            - The logit model is used to probabilistically select the next node among 
              valid neighbors.
            - The method returns None if the maximum path length is exceeded or 
              if no valid options are available at any step.
        """
    
        path, current_node = list(), origin
        while True:
            path.append(current_node)
            
            if self.allow_loops:
                options = sorted(self.network.neighbors(current_node))
            else:
                options = [node for node in sorted(self.network.neighbors(current_node)) if (node not in path)]
                
            if   (destination in options):                  return path + [destination]
            elif (not options) or (len(path) > self.max_path_length):     return None
            else:       
                try:            
                    current_node = self._logit(options, node_potentials)
                except:
                    return None
    
    
    def _shift_parameters(self, tmp_beta: float, tmp_max_path_length: Union[int, float]):
        if self.params_to_shift in ["beta", "both"]:
            shifted_beta = tmp_beta * ((100-self.shift_parameters_by) / 100) # Increase beta by shift_params%
            tmp_beta = min(shifted_beta, -.01)
        if (self.params_to_shift in ["max_path_length", "both"]) and (tmp_max_path_length != float('inf')):
            shifted_max_path_len = tmp_max_path_length * ((100+self.shift_parameters_by) / 100)
            tmp_max_path_length = int(shifted_max_path_len) # Increase max_path_length by shift_params%
        return tmp_beta, tmp_max_path_length
    
    
    def _pick_routes_from_samples(self, sampled_routes: list[tuple]) -> list[tuple]:
        return super()._pick_routes_from_samples(sampled_routes)


    def _logit(self, options: list, node_potentials: dict) -> str:
        return super()._logit(options, node_potentials)