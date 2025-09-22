import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import lxml
import networkx as nx
import pandas as pd
from pathlib import Path

from bs4 import BeautifulSoup
from typing import Tuple

from janux.utils import remove_double_quotes


#################################################

def build_digraph(connection_file: str, edge_file: str, route_file: str) -> nx.DiGraph:
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
    """
    try:
        # Process connections
        connections_df = _process_connection_file(connection_file)

        # Process edge attributes
        edge_attributes_df = _process_edge_file(edge_file)

        # Process route attributes
        route_path = Path(route_file) ## maybe we can add the .net file as argument in this function 
                                       ## (I did not do it to not to pass new arguments in the simulator)
        net_file = route_path.with_name(route_path.stem.replace(".rou", ".net") + route_path.suffix)

        route_attributes_df = _process_route_file(route_file,net_file)

        # Merge all DataFrames and calculate travel times
        network_df = _merge_network_data(connections_df, edge_attributes_df, route_attributes_df)
        network_df = network_df.mask(network_df.astype(object).eq('None')).dropna()

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
    
#################################################


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


def _process_route_file(route_file: str, net_file: str) -> pd.DataFrame:
    """Parses the route XML file and returns a DataFrame with route attributes."""
    with open(route_file, 'r') as route_file_obj:
        route_xml_data = route_file_obj.read()
    route_xml_parsed = BeautifulSoup(route_xml_data, "xml")

    if route_xml_parsed.find('edge'): # Some .rou files contain the <edge> element 
        edges = route_xml_parsed.find_all('edge', {'to': True})

        # Extract attributes from route XML
        route_attributes_df = pd.DataFrame({
            'edge_id': [edge.get('id') for edge in edges],
            'length': [edge.find('lane').get('length') for edge in edges],
            'speed': [edge.find('lane').get('speed') for edge in edges]
        })
    else:  # Some .rou files are missing this element and the additional info (edge_id, length, speed) are included in the .net file 
       route_attributes_df = _process_route_and_net(route_file, net_file)
    
    return route_attributes_df

def _process_route_and_net(route_file: str, net_file: str) -> pd.DataFrame:
    """
    Parses a SUMO .rou(.xml) file and a .net.xml file and returns a DataFrame
    with edge attributes for all edges that appear in any vehicle's route.

    Returns columns: edge_id, length, speed (as strings from .net; cast if needed).
    """
    # --- Parse the route file ---
    with open(route_file, 'r', encoding='utf-8') as f:
        rou_xml = BeautifulSoup(f.read(), "xml")

    edge_ids = set()

    # 1) <vehicle><route edges="e1 e2 ..."/></vehicle>
    for v in rou_xml.find_all('vehicle'):
        r = v.find('route')
        if r and r.has_attr('edges'):
            edge_ids.update(r['edges'].split())

    # 2) <vehicle><route> <edge id="..."/>* </route></vehicle>
        if r and not r.has_attr('edges'):
            for e in r.find_all('edge'):
                if e.has_attr('id'):
                    edge_ids.add(e['id'])

    # 3) Named routes: <route id="r1" edges="..."> or with child <edge id="..."/>
    for r in rou_xml.find_all('route'):
        # edges="..."
        if r.has_attr('edges'):
            edge_ids.update(r['edges'].split())
        else:
            # child edges
            for e in r.find_all('edge'):
                if e.has_attr('id'):
                    edge_ids.add(e['id'])

    # 4) Vehicles referencing a named route: <vehicle route="r1">
    for v in rou_xml.find_all('vehicle'):
        if v.has_attr('route'):
            rid = v['route']
            r = rou_xml.find('route', {'id': rid})
            if r:
                if r.has_attr('edges'):
                    edge_ids.update(r['edges'].split())
                else:
                    for e in r.find_all('edge'):
                        if e.has_attr('id'):
                            edge_ids.add(e['id'])

    # --- Parse the net file to get length/speed per edge ---
    with open(net_file, 'r', encoding='utf-8') as f:
        net_xml = BeautifulSoup(f.read(), "xml")

    # Build lookup: edge_id -> (length, speed). Use the first lane if multiple.
    length_lookup = {}
    speed_lookup = {}
    for ed in net_xml.find_all('edge', {'id': True}):
        eid = ed['id']
        lane = ed.find('lane')  # first lane
        if lane:
            length_lookup[eid] = lane.get('length')
            speed_lookup[eid]  = lane.get('speed')

    # Assemble rows for edges that appear in routes (keep only those we can resolve)
    rows = []
    for eid in sorted(edge_ids):
        length = length_lookup.get(eid)
        speed = speed_lookup.get(eid)
        rows.append({'edge_id': eid, 'length': length, 'speed': speed})

    df = pd.DataFrame(rows)

    # Optional: cast to numeric (keeps NaN if an edge wasn't found in the net)
    for col in ['length', 'speed']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


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