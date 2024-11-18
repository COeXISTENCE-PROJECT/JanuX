import networkx as nx
import numpy as np
import os
import pandas as pd
import sys
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

from mock_keychain import Keychain as kc
from utils import get_params
from utils import remove_double_quotes
from utils import df_to_prettytable

from generate_paths import generate_paths


############# Network Generation ################

def generate_network(connection_file: str, edge_file: str, route_file: str) -> nx.DiGraph:
    """
    Generates a traffic network graph from XML files.

    Args:
        connection_file (str): Path to the connection XML file.
        edge_file (str): Path to the edge XML file.
        route_file (str): Path to the route XML file.

    Returns:
        nx.DiGraph: A directed graph representing the traffic network.

    Raises:
        FileNotFoundError: If any of the input files are not found.
        ValueError: If required data is missing in the input files.

    Example:
        >>> network = generate_network("connections.xml", "edges.xml", "routes.xml")
        >>> type(network)
        <class 'networkx.DiGraph'>
    """
    try:
        # Process connections
        connections_df = _process_connection_file(connection_file)

        # Process edge attributes
        edge_attributes_df = _process_edge_file(edge_file)

        # Process route attributes
        route_attributes_df = _process_route_file(route_file)

        # Merge all DataFrames and calculate travel times
        network_df = _merge_network_data(connections_df, edge_attributes_df, route_attributes_df)

        # Create and return directed graph
        traffic_network_graph = nx.from_pandas_edgelist(
            network_df,
            source='source_edge',
            target='target_edge',
            edge_attr='travel_time',
            create_using=nx.DiGraph()
        )
        return traffic_network_graph

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def _process_connection_file(connection_file: str) -> pd.DataFrame:
    """Parses the connection XML file and returns a DataFrame."""
    from_df, to_df = _read_xml_file(connection_file, 'connection', 'from', 'to')
    connections_df = pd.merge(from_df, to_df, left_index=True, right_index=True)
    connections_df = connections_df.rename(columns={'0_x': 'source_edge', '0_y': 'target_edge'})
    return connections_df


def _process_edge_file(edge_file: str) -> pd.DataFrame:
    """Parses the edge XML file and returns a DataFrame with edge attributes."""
    edge_ids_df, edge_sources_df = _read_xml_file(edge_file, 'edge', 'id', 'from')
    edge_attributes_df = pd.merge(edge_sources_df, edge_ids_df, right_index=True, left_index=True)
    edge_attributes_df = edge_attributes_df.rename(columns={'0_x': 'source_node', '0_y': 'edge_id'})
    edge_attributes_df['source_node'] = edge_attributes_df['source_node'].apply(remove_double_quotes)
    edge_attributes_df['edge_id'] = edge_attributes_df['edge_id'].apply(remove_double_quotes)
    return edge_attributes_df


def _process_route_file(route_file: str) -> pd.DataFrame:
    """Parses the route XML file and returns a DataFrame with route attributes."""
    with open(route_file, 'r') as route_file_obj:
        route_xml_data = route_file_obj.read()
    route_xml_parsed = BeautifulSoup(route_xml_data, "xml")
    edges = route_xml_parsed.find_all('edge', {'to': True})

    # Extract attributes from route XML
    route_attributes_df = pd.DataFrame({
        'edge_id': [edge.get('id') for edge in edges],
        'length': [edge.find('lane').get('length') for edge in edges],
        'speed': [edge.find('lane').get('speed') for edge in edges]
    })
    return route_attributes_df


def _merge_network_data(
    connections_df: pd.DataFrame,
    edge_attributes_df: pd.DataFrame,
    route_attributes_df: pd.DataFrame
) -> pd.DataFrame:
    """Merges connection, edge, and route DataFrames and calculates travel times."""
    edge_route_merged_df = pd.merge(edge_attributes_df, route_attributes_df, on='edge_id', how='inner')
    network_df = pd.merge(edge_route_merged_df, connections_df, left_on='edge_id', right_on='source_edge', how='inner')

    # Calculate travel time (in minutes)
    network_df['travel_time'] = (network_df['length'].astype(float) / network_df['speed'].astype(float)) / 60
    network_df = network_df[['source_edge', 'target_edge', 'travel_time']]  # Keep only required columns
    return network_df


def _read_xml_file(file_path: str, element_name: str, attr1: str, attr2: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads an XML file and extracts two specified attributes."""
    with open(file_path, 'r') as f:
        data = f.read()
    parsed_xml = BeautifulSoup(data, "xml")
    elements = parsed_xml.find_all(element_name)

    attr1_values = [el.get(attr1) for el in elements]
    attr2_values = [el.get(attr2) for el in elements]

    return pd.DataFrame(attr1_values), pd.DataFrame(attr2_values)

#################################################



####################### Main #######################

if __name__ == "__main__":
    params = get_params("test/mock_params.json")
    gen_params = params[kc.PATH_GEN]

    origins = gen_params[kc.ORIGINS]
    destinations = gen_params[kc.DESTINATIONS]

    sim_params = params[kc.SIMULATOR]
    connection_file_path = "test/" + sim_params[kc.CONNECTION_FILE_PATH]
    edge_file_path = "test/" + sim_params[kc.EDGE_FILE_PATH]
    route_file_path = "test/" + sim_params[kc.ROUTE_FILE_PATH]

    network = generate_network(connection_file_path, edge_file_path, route_file_path)
    routes = generate_paths(network, origins, destinations, num_samples=100, number_of_paths=3)
    #df_to_prettytable(routes)
    routes.to_csv("test/routes.csv", index=False)
    

#################################################