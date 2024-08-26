from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_caching import Cache
import networkx as nx
import pandas as pd
import numpy as np
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure Flask-Caching
app.config['CACHE_TYPE'] = 'simple'  # Use simple cache for development
cache = Cache(app)

# Global graph variable
G = None

def initialize_graph():
    """Initialize the global graph G from CSV files."""
    global G
    nodes_df = pd.read_csv('synthetic_nodes.csv')
    edges_df = pd.read_csv('synthetic_edges.csv')

    G = nx.Graph()

    for _, row in nodes_df.iterrows():
        G.add_node(row['ID'], label=row['Label'])

    for _, row in edges_df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row.get('Weight', 1))

def calculate_positions():
    """Calculate 3D positions for nodes."""
    return nx.spring_layout(G, dim=3, seed=42)

def get_edge_metrics():
    """Calculate various edge metrics."""
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    scaling_factor = 1000
    scaled_centrality = {node: centrality * scaling_factor for node, centrality in betweenness_centrality.items()}

    return betweenness_centrality, scaled_centrality

def get_thresholds(centrality_values):
    """Calculate percentile thresholds for color coding."""
    return {
        'green': np.percentile(centrality_values, 90),
        'aqua': np.percentile(centrality_values, 75)
    }

def get_color_by_percentile(centrality, thresholds):
    """Determine node color based on centrality percentile thresholds."""
    if centrality >= thresholds['green']:
        return '#00ff00'
    elif centrality >= thresholds['aqua']:
        return '#00c0ff'
    else:
        return '#ff2e89'

def get_size(degree, max_degree, min_size=5, max_size=20):
    """Calculate node size based on degree centrality."""
    return min_size + (degree / max_degree) * (max_size - min_size)

def calculate_spokes(G):
    """Calculate the number of spokes for each node."""
    degree_dict = dict(G.degree())
    spokes_dict = {}
    for node in G.nodes():
        # Count the number of neighboring nodes with degree 1
        spokes = sum(1 for neighbor in G.neighbors(node) if degree_dict[neighbor] == 1)
        spokes_dict[node] = spokes
    return spokes_dict

def filter_edges(G, strategy='betweenness', threshold=0.1):
    """Filter edges based on the selected strategy and remove isolated nodes."""
    if strategy == 'betweenness':
        return filter_edges_by_betweenness(G, threshold)
    elif strategy == 'frequency':
        return filter_edges_by_frequency(G, threshold)
    elif strategy == 'information':
        return filter_edges_by_information(G, threshold)
    elif strategy == 'random':
        return filter_edges_randomly(G, threshold)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def filter_edges_by_betweenness(G, threshold):
    edge_betweenness = nx.edge_betweenness_centrality(G)
    edges_sorted = sorted(edge_betweenness.items(), key=lambda item: item[1], reverse=True)
    to_remove = int(len(edges_sorted) * threshold)
    G.remove_edges_from([edge for edge, _ in edges_sorted[:to_remove]])
    remove_isolated_nodes(G)  # Remove isolated nodes
    return G

def filter_edges_by_frequency(G, threshold):
    edges_sorted = sorted(G.edges(data=True), key=lambda edge: edge[2].get('weight', 1), reverse=True)
    to_remove = int(len(edges_sorted) * threshold)
    G.remove_edges_from([edge[:2] for edge in edges_sorted[:to_remove]])
    remove_isolated_nodes(G)  # Remove isolated nodes
    return G

def filter_edges_by_information(G, threshold):
    edge_information = {}
    total_weight = sum(d['weight'] for _, _, d in G.edges(data=True))
    for u, v, data in G.edges(data=True):
        p_uv = data['weight'] / total_weight
        p_u = sum(d['weight'] for x, y, d in G.edges(data=True) if x == u or y == u) / total_weight
        p_v = sum(d['weight'] for x, y, d in G.edges(data=True) if x == v or y == v) / total_weight
        mutual_info = p_uv * np.log(p_uv / (p_u * p_v)) if p_u * p_v > 0 else 0
        edge_information[(u, v)] = mutual_info
    edges_sorted = sorted(edge_information.items(), key=lambda item: item[1], reverse=True)
    to_remove = int(len(edges_sorted) * threshold)
    G.remove_edges_from([edge for edge, _ in edges_sorted[:to_remove]])
    remove_isolated_nodes(G)  # Remove isolated nodes
    return G

def filter_edges_randomly(G, threshold):
    edges = list(G.edges())
    random.shuffle(edges)
    to_remove = int(len(edges) * threshold)
    G.remove_edges_from(edges[:to_remove])
    remove_isolated_nodes(G)  # Remove isolated nodes
    return G

def remove_isolated_nodes(G):
    """Remove nodes that no longer have any edges."""
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

@app.route('/process_graph', methods=['GET'])
@cache.cached(timeout=3600, key_prefix='process_graph')
def process_graph():
    """Handle GET request to process and return graph data."""
    global G
    initialize_graph()
    pos_3d = calculate_positions()
    pos_3d_dict = {node: pos_3d[node].tolist() for node in G.nodes()}

    betweenness_centrality, scaled_centrality = get_edge_metrics()
    centrality_values = list(scaled_centrality.values())
    thresholds = get_thresholds(centrality_values)

    degree_dict = dict(G.degree())
    max_degree = max(degree_dict.values(), default=1)

    # Calculate spokes
    spokes_dict = calculate_spokes(G)

    response_data = {
        'nodes': [
            {
                'id': node,
                'x': pos_3d_dict[node][0],
                'y': pos_3d_dict[node][1],
                'z': pos_3d_dict[node][2],
                'degree_centrality': scaled_centrality[node],
                'color': get_color_by_percentile(scaled_centrality[node], thresholds),
                'size': get_size(degree_dict[node], max_degree),
                'degree': degree_dict[node],  # Number of connections
                'spokes': spokes_dict[node]  # Number of spokes
            } for node in G.nodes()
        ],
        'edges': [
            {
                'source': edge[0],
                'target': edge[1],
                'source_pos': pos_3d_dict[edge[0]],
                'target_pos': pos_3d_dict[edge[1]]
            } for edge in G.edges()
        ],
        'hubs': []
    }

    return jsonify(response_data)

@app.route('/filter_edges', methods=['OPTIONS', 'POST'])
def filter_edges_route():
    """Handle POST request to filter edges based on the strategy."""
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight request successful'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response

    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON payload.'}), 400

            strategy = data.get('strategy', 'betweenness')
            threshold = float(data.get('threshold', 0.1))

            global G
            if G is None:
                return jsonify({'error': 'Graph not initialized.'}), 400

            # Apply edge filtering
            G = filter_edges(G, strategy=strategy, threshold=threshold)

            # Generate graph data
            pos_3d = calculate_positions()
            pos_3d_dict = {node: pos_3d[node].tolist() for node in G.nodes()}

            betweenness_centrality, scaled_centrality = get_edge_metrics()
            centrality_values = list(scaled_centrality.values())
            thresholds = get_thresholds(centrality_values)

            degree_dict = dict(G.degree())
            max_degree = max(degree_dict.values(), default=1)

            # Calculate spokes
            spokes_dict = calculate_spokes(G)

            response_data = {
                'nodes': [
                    {
                        'id': node,
                        'x': pos_3d_dict[node][0],
                        'y': pos_3d_dict[node][1],
                        'z': pos_3d_dict[node][2],
                        'degree_centrality': scaled_centrality[node],
                        'color': get_color_by_percentile(scaled_centrality[node], thresholds),
                        'size': get_size(degree_dict[node], max_degree),
                        'degree': degree_dict[node],  # Number of connections
                        'spokes': spokes_dict[node]  # Number of spokes
                    } for node in G.nodes()
                ],
                'edges': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'source_pos': pos_3d_dict[edge[0]],
                        'target_pos': pos_3d_dict[edge[1]]
                    } for edge in G.edges()
                ],
                'hubs': []
            }

            return jsonify(response_data)

        except Exception as e:
            print(e)
            return jsonify({'error': 'Error processing request.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
