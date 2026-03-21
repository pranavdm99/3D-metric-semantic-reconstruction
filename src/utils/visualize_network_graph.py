#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    import subprocess
    print("Installing networkx...")
    subprocess.check_call(['pip', 'install', 'networkx'])
    import networkx as nx

def plot_network_graph():
    data_root = Path("/workspace/data")
    json_path = data_root / "outputs" / "masks" / "scene_graph.json"
    output_path = data_root / "outputs" / "masks" / "network_graph_map.png"
    
    if not json_path.exists():
        json_path = Path("data/outputs/masks/scene_graph.json")
        output_path = Path("data/outputs/masks/scene_graph_map.png")
        if not json_path.exists():
            print(f"❌ Error: {json_path} not found.")
            return

    print(f"📖 Loading Scene Graph from {json_path}...")
    with open(json_path, 'r') as f:
        graph_data = json.load(f)
        
    G = nx.DiGraph()
    
    for obj_name in graph_data['objects'].keys():
        G.add_node(obj_name)
        
    edge_labels = {}
    
    print("🕸️ Building Semantic Graph Edges...")
    for rel in graph_data.get('relationships', []):
        subj = rel['subject']
        obj = rel['object']
        preds = rel['predicates']
        
        if not preds: continue
        
        # We want to extract meaningful topological predicates to avoid a massive hairball.
        # "across_room" and cardinal directions are too dense for a clean topology graph.
        
        label_parts = []
        if 'resting_on' in preds:
            label_parts.append('rests on')
        elif 'directly_above' in preds:
            label_parts.append('above')
        elif 'directly_below' in preds:
            pass # redundant if we have 'above'
            
        if 'next_to' in preds and not label_parts:
            label_parts.append('next to')
            
        if not label_parts:
            continue
            
        label = "\n".join(label_parts)
        
        if label:
            G.add_edge(subj, obj)
            edge_labels[(subj, obj)] = label

    plt.figure(figsize=(16, 12))
    
    # Use spring layout but with a lot of spacing
    pos = nx.spring_layout(G, k=2.0, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='#lightblue'[0] if False else 'lightblue', edgecolors='black', linewidths=2)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=30, edge_color='gray', width=2, node_size=3000, connectionstyle='arc3,rad=0.0')
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5, font_color='darkred', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.5))
    
    plt.title("Semantic Relationship Network Grid\n(Arrows point from Subject to Object)", fontsize=20, pad=20)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Semantic network graph rendered to: {output_path}")

if __name__ == "__main__":
    plot_network_graph()
