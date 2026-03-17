#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt

def visualize(config_path: str):
    # Load config and metadata
    data_root = Path("/workspace/data")
    meta_path = data_root / "outputs" / "masks" / "segmentation_metadata.json"
    
    if not meta_path.exists():
        print(f"❌ Error: Metadata not found at {meta_path}")
        return

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    splat_path = Path(meta['splat_path'])
    if not splat_path.exists():
        # Try local path if workspace path fails
        splat_path = data_root / "outputs" / "splats" / "sugar_output" / "sugar_refined.ply"

    print(f"📖 Loading splats from {splat_path}...")
    plydata = PlyData.read(str(splat_path))
    vertices = plydata['vertex']
    n_pts = len(vertices)
    
    # Create new color arrays
    red = np.zeros(n_pts, dtype=np.uint8) + 50  # Dim gray for background
    green = np.zeros(n_pts, dtype=np.uint8) + 50
    blue = np.zeros(n_pts, dtype=np.uint8) + 50
    
    # Unique colors per object
    objects = meta['objects']
    colormap = plt.get_cmap('tab20')
    
    print(f"🎨 Coloring {len(objects)} objects...")
    for i, obj_name in enumerate(objects):
        indices = meta['gaussian_indices'].get(obj_name, [])
        if not indices: continue
        
        color = np.array(colormap(i % 20)[:3]) * 255
        red[indices] = int(color[0])
        green[indices] = int(color[1])
        blue[indices] = int(color[2])
        print(f"  → {obj_name}: {len(indices):,} points (Color: {color.astype(int)})")

    # Create new PLY elements
    # We keep x, y, z and add/override red, green, blue
    new_data = []
    for name in vertices.data.dtype.names:
        if name in ['red', 'green', 'blue', 'colors']: continue
        new_data.append((name, vertices[name]))
        
    new_vertices = np.empty(n_pts, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    new_vertices['x'] = vertices['x']
    new_vertices['y'] = vertices['y']
    new_vertices['z'] = vertices['z']
    new_vertices['red'] = red
    new_vertices['green'] = green
    new_vertices['blue'] = blue
    
    output_path = data_root / "outputs" / "masks" / "segmented_scene.ply"
    print(f"💾 Saving visualization to {output_path}...")
    PlyData([PlyElement.describe(new_vertices, 'vertex')]).write(str(output_path))
    print(f"✅ Done! Open this file in MeshLab or CloudCompare to see the segments.")

if __name__ == "__main__":
    visualize("/workspace/configs/pipeline_config.json")
