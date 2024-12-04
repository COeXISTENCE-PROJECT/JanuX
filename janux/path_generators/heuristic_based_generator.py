import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import inspect
import networkx as nx
import numpy as np
import pandas as pd

from typing import Callable, List, Union

from janux.path_generators import calculate_free_flow_time
from janux.path_generators import paths_to_df
from janux.path_generators.extended_generator import ExtendedPathGenerator
from janux.utils import iterable_to_string

class HeuristicPathGenerator(ExtendedPathGenerator):
    
    """
    Generates paths in a transportation network using heuristic-based scoring.

    This class extends the functionality of `ExtendedPathGenerator` by incorporating heuristics 
    to evaluate and select optimal paths from a set of sampled routes. Heuristics are customizable 
    and allow the user to guide the selection process based on custom criteria.

    Args:
        network (nx.DiGraph): The transportation network represented as a directed graph.
        origins (list[str]): A list of origin nodes in the network.
        destinations (list[str]): A list of destination nodes in the network.
        heuristics (list[Callable]): A list of heuristic functions to evaluate route groups. 
                                     Each heuristic must accept `number_of_paths + 1` arguments 
                                     (paths and the network) and return a numerical score.
        heur_weights (list[float]): A list of weights for the heuristics. Must match the number 
                                    of heuristics provided.
        **kwargs: Additional parameters for route generation, including those inherited from 
                  `ExtendedPathGenerator`.

    Attributes:
        heuristics (list[Callable]): List of heuristic functions used to evaluate route groups.
        heur_weights (list[float]): Weights assigned to each heuristic during route evaluation.

    Methods:
        generate_routes(as_df=True): Generates routes between origin-destination pairs, scoring 
                                     and selecting optimal paths using the provided heuristics.
        _pick_routes_from_samples(sampled_routes): Selects the most optimal set of routes from 
                                                   sampled routes using heuristic scoring.
        _validate_heuristics(sampled_routes): Validates the heuristics to ensure they are callable, 
                                              deterministic, and produce numerical results.
    """
    
    def __init__(self, 
                 network: nx.DiGraph, 
                 origins: list[str], 
                 destinations: list[str], 
                 heuristics: list[Callable],
                 heur_weights: list[float],
                 **kwargs):
        
        super().__init__(network, origins, destinations, **kwargs)
        
        self.heuristics = heuristics
        self.heur_weights = heur_weights
        
        
    def generate_routes(self, as_df: bool = True, calc_free_flow: bool = False) -> Union[pd.DataFrame, dict]:
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
                sampled_routes = set()   # num_samples number of routes
                iteration_count = 0
                initial_beta, initial_max_path_len = self.beta, self.max_path_length
                while (len(sampled_routes) < self.num_samples):
                    
                    # If this gets stuck, increase beta and max_path_length
                    if (self.adaptive) and (iteration_count > self.tolerate_num_iterations):
                        self.logger.warning(f"Exceeded tolerance for {origin_idx} -> {dest_idx}.")
                        self.beta, self.max_path_length = self._shift_parameters(self.beta, self.max_path_length)
                        self.logger.info(f"Beta: {self.beta}, Max Path Length: {self.max_path_length}")
                        iteration_count = 0
                        
                    path = self._sample_single_route(origin_name, dest_name, node_potentials)
                    if not path is None:
                        sampled_routes.add(tuple(path))
                    iteration_count += 1
                
                self.beta, self.max_path_length = initial_beta, initial_max_path_len  
                self.logger.info(f"Sampled {len(sampled_routes)} paths for {origin_idx} -> {dest_idx}")
                sampled_routes = sorted(list(sampled_routes), key=lambda x: iterable_to_string(x))
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


    def _pick_routes_from_samples(self, sampled_routes: list[tuple]) -> list[tuple]:
        """
        Selects the most optimal set of routes from a list of sampled routes using heuristic scoring.

        This method evaluates groups of sampled routes based on a set of heuristics and their respective
        weights. It generates multiple groups of routes, calculates their scores using the heuristics,
        and selects the group with the highest overall score.

        Args:
            sampled_routes (list[tuple]): A list of sampled routes, where each route is represented 
                                        as a tuple of nodes.

        Returns:
            list[tuple]: A list of the selected routes, where each route is represented as a tuple of nodes.

        Raises:
            AssertionError:
                - If the desired number of paths to select exceeds the number of sampled routes.
                - If the number of paths to select (`number_of_paths`) is less than or equal to zero.
                - If any heuristic is not callable, does not accept the expected number of arguments,
                does not return a numerical value, or is non-deterministic.
                - If the number of heuristic weights does not match the number of heuristics.
        """
        assert self.number_of_paths <= len(sampled_routes), f"Number of paths ({self.number_of_paths}) should be less than or equal to the number of sampled routes ({len(sampled_routes)})"
        assert self.number_of_paths > 0, f"Number of paths should be greater than 0"
        self._validate_heuristics(sampled_routes)
        
        # Sample groups of routes (as their indices in the sampled_routes list)
        route_group_samples = set()
        while len(route_group_samples) < self.num_samples:
            sampled_group = self.rng.choice(len(sampled_routes), size=self.number_of_paths, replace=False)
            sampled_group = tuple(sorted(sampled_group))
            route_group_samples.add(sampled_group)
        route_group_samples = list(route_group_samples)
        
        # Calculate scores for each group of sampled paths
        scores = list()
        for sampled_group in route_group_samples:
            score = 0
            paths = [sampled_routes[i] for i in sampled_group]
            for heur_weight, heuristic in zip(self.heur_weights, self.heuristics):
                score += heur_weight * heuristic(*paths, self.network)
            scores.append(score)
        
        # Select the group of sampled paths with the highest score
        selected_paths = route_group_samples[np.argmax(scores)]
        selected_paths = [sampled_routes[i] for i in selected_paths]
        return selected_paths
    

    def _validate_heuristics(self, sampled_routes: list[tuple]) -> None:
        # Assert that each heuristic accepts as argument number_of_paths+1 items (paths and network)
        for heuristic in self.heuristics:
            # Check if the heuristic is callable
            assert callable(heuristic), f"Each heuristic must be callable, but found: {type(heuristic)}"
            # Check the number of arguments
            num_arguments = len(inspect.signature(heuristic).parameters)
            assert num_arguments == self.number_of_paths+1, f"Each heuristic must accept exactly {self.number_of_paths+1} arguments, but {heuristic.__name__} takes {num_arguments}."
            # Check if return is numerical
            example_return = heuristic(*sampled_routes[:self.number_of_paths], self.network)
            assert isinstance(example_return, (int, float)), f"Each heuristic must return a numerical value, but {heuristic.__name__} returns {type(example_return)}"
            # Check if the heuristic is deterministic
            assert heuristic(*sampled_routes[:self.number_of_paths], self.network) == heuristic(*sampled_routes[:self.number_of_paths], self.network), f"Each heuristic must be deterministic, but {heuristic.__name__} is not."
        assert len(self.heur_weights) == len(self.heuristics), f"Number of heuristic weights does not match with number of heuristics. ({len(self.heur_weights)} and {len(self.heuristics)})"
        
    
    def _sample_single_route(self, origin: str, destination: str, node_potentials: dict) -> Union[List[str], None]:
        return super()._sample_single_route(origin, destination, node_potentials)