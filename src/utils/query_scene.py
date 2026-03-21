#!/usr/bin/env python3
import json
import argparse
import requests
from pathlib import Path

def query_scene(query_text="What is resting on the workdesk?", model="qwen3-vl:8b-instruct"):
    data_root = Path("/workspace/data")
    graph_path = data_root / "outputs" / "masks" / "scene_graph.json"
    
    if not graph_path.exists():
        # Fallback to host path if not exactly inside docker
        graph_path = Path("data/outputs/masks/scene_graph.json")
        if not graph_path.exists():
            print(f"Error: {graph_path} not found. Did you run generate_scene_graph.py?")
            return

    with open(graph_path, 'r') as f:
        graph = json.load(f)

    # Simplify decimal precision and structure to avoid overwhelming the LLM prompt
    simplified_objects = {}
    for name, obj in graph['objects'].items():
        simplified_objects[name] = {
            "centroid": {k: round(v, 2) for k, v in obj['centroid'].items()},
            "dimensions_meters": {k: round(v, 2) for k, v in obj['dimensions'].items()}
        }
        
    simplified_graph = {
        "spatial_objects": simplified_objects,
        "pairwise_relationships": [
            {
                "A": r['subject'],
                "B": r['object'],
                "distance": round(r['distance_meters'], 2),
                "is_active": [p for p in r['predicates']]
            } for r in graph['relationships'] if r['predicates']
        ]
    }

    system_prompt = (
        "You are a physical reasoning AI analyzing a metric 3D semantic scene graph. "
        "The coordinate system is in meters, and the Y-axis points straight UP (against gravity). "
        "CRITICAL INSTRUCTION: You MUST base your answer strictly on the specific objects given in the ### SCENE GRAPH ### below. "
        "If the user asks what is in a certain direction, DO NOT just explain the coordinate system. You MUST look at the X, Y, Z centroids of the objects in the JSON and list the actual object names (e.g., 'workdesk_0') that exist in those directions. "
        "Answer naturally but reference exact metrics (like dimensions or distances) to prove you are grounded in the data."
        f"\n\n### SCENE GRAPH ###\n{json.dumps(simplified_graph, indent=2)}\n"
    )

    print(f"🤖 Sending query: '{query_text}'")
    
    try:
        # Assumes Ollama is running on localhost exposing the inference API
        # Or host.docker.internal if running from inside Docker container
        host = "host.docker.internal"
        try:
            requests.get(f"http://{host}:11434/", timeout=1)
        except requests.exceptions.RequestException:
            host = "localhost" # Fallback
            
        ollama_url = f"http://{host}:11434/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": f"{system_prompt}\n\nUSER QUESTION: {query_text}"}
            ],
            "stream": False
        }
        
        response = requests.post(ollama_url, json=payload, timeout=300)
        
        if response.status_code == 200:
            content = response.json().get('message', {}).get('content', '')
            print("\n" + "="*50)
            print(content.strip())
            print("="*50 + "\n")
        else:
            print(f"❌ LLM API Error: {response.status_code} - {response.text}")
            print("💡 Reminder: Ensure 'ollama serve' is running and the model is downloaded!")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to the local LLM on {host}:11434.")
        print("Detailed Error:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the 3D scene graph using Natural Language")
    parser.add_argument("query", nargs="?", help="The question to ask about the scene")
    parser.add_argument("--model", type=str, default="qwen3-vl:8b-instruct", help="The Ollama model to use")
    args = parser.parse_args()
    query_scene(args.query, args.model)
