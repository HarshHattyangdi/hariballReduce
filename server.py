from flask import Flask, jsonify
import pandas as pd
import networkx as nx
import numpy as np
from flask_cors import CORS
from flask_caching import Cache

app = Flask(__name__)
CORS(app)

# Configure Flask-Caching
app.config['CACHE_TYPE'] = 'simple'  # Use simple cache for development
cache = Cache(app)

@app.route('/process_graph', methods=['GET'])
@cache.cached(timeout=3600, key_prefix='process_graph')  # Cache response for 1 hour
def process_graph():
    # Load data from files
    nodes_df = pd.read_csv('synthetic_nodes.csv')
    edges_df = pd.read_csv('synthetic_edges.csv')

    # Create the NetworkX graph
    G = nx.Graph()
    
    # Add nodes and edges
    for _, row in nodes_df.iterrows():
        G.add_node(row['ID'], label=row['Label'])

    for _, row in edges_df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row.get('Weight', 1))

    # Compute positions using a layout algorithm
    pos_3d = nx.spring_layout(G, dim=3, seed=42)
    pos_3d_dict = {node: pos_3d[node].tolist() for node in G.nodes()}

    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    scaling_factor = 1000  # Larger factor to make the values more visible
    scaled_centrality = {node: centrality * scaling_factor for node, centrality in betweenness_centrality.items()}

    # Normalize scaled centrality to range [0, 1] for color mapping
    max_centrality = max(scaled_centrality.values(), default=1)  # Avoid division by zero
    normalized_centrality = {node: centrality / max_centrality for node, centrality in scaled_centrality.items()}

    # Identify hubs using a percentile-based threshold
    degree_sequence = sorted(dict(G.degree()).values(), reverse=True)
    threshold_index = int(len(degree_sequence) * 0.05)  # Top 5% of nodes by degree
    hub_threshold = degree_sequence[threshold_index] if threshold_index < len(degree_sequence) else degree_sequence[-1]

    hubs = [node for node, degree in dict(G.degree()).items() if degree >= hub_threshold]

    # Prepare hub-and-spoke data
    hub_and_spokes = []
    for hub in hubs:
        spokes = [n for n in G.neighbors(hub) if G.degree(n) < hub_threshold]
        if spokes:  # Only include hubs that have spokes
            hub_and_spokes.append({
                'hub': {
                    'id': hub,
                    'x': pos_3d_dict[hub][0],
                    'y': pos_3d_dict[hub][1],
                    'z': pos_3d_dict[hub][2],
                    'degree_centrality': scaled_centrality[hub]
                },
                'spokes': [
                    {
                        'id': spoke,
                        'x': pos_3d_dict[spoke][0],
                        'y': pos_3d_dict[spoke][1],
                        'z': pos_3d_dict[spoke][2],
                        'degree_centrality': scaled_centrality[spoke]
                    } for spoke in spokes
                ]
            })

    # Determine node colors based on normalized degree centrality
    def get_color(normalized_degree):
        if normalized_degree > 0.7:
            return '#00ff00'  # Green for high degree
        elif normalized_degree > 0.4:
            return '#ffff00'  # Yellow for medium degree
        else:
            return '#ff0000'  # Red for low degree

    # Prepare response data with degree centrality, color, and hubs
    response_data = {
        'nodes': [
            {
                'id': node,
                'x': pos_3d_dict[node][0],
                'y': pos_3d_dict[node][1],
                'z': pos_3d_dict[node][2],
                'degree_centrality': scaled_centrality[node],
                'color': get_color(normalized_centrality[node])
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
        'hubs': hub_and_spokes  # Include hub-and-spoke data
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
