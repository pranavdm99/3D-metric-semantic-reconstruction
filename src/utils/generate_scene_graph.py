#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from plyfile import PlyData

def filter_outliers(points, percentile=5):
    """Remove extreme outliers to accurately estimate the metric bounding box."""
    if len(points) == 0:
        return points
    
    # Calculate bounds at specified percentiles
    lower = np.percentile(points, percentile, axis=0)
    upper = np.percentile(points, 100 - percentile, axis=0)
    
    # Filter points within bounds
    mask = np.all((points >= lower) & (points <= upper), axis=1)
    return points[mask]

def compute_xz_obb(points):
    """Compute an Oriented Bounding Box aligned with Y (gravity) but oriented freely in XZ."""
    centroid = np.mean(points, axis=0)
    
    # Y-axis bounds (Gravity / Height)
    min_y = float(np.min(points[:, 1]))
    max_y = float(np.max(points[:, 1]))
    height_y = max_y - min_y
    
    # XZ plane PCA for orientation
    xz_pts = points[:, [0, 2]]
    xz_centroid = centroid[[0, 2]]
    centered_xz = xz_pts - xz_centroid
    
    # Covariance matrix of XZ
    cov = np.cov(centered_xz, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    
    # Sort principal axes (largest variance first)
    sort_indices = np.argsort(evals)[::-1]
    evecs = evecs[:, sort_indices]
    
    # Ensure right-handed XZ basis
    if np.linalg.det(evecs) < 0:
        evecs[:, 1] = -evecs[:, 1]
    
    # Project XZ points to find width/depth
    projected_xz = centered_xz @ evecs
    min_xz = np.min(projected_xz, axis=0)
    max_xz = np.max(projected_xz, axis=0)
    
    dimensions_xz = max_xz - min_xz
    width_x = float(dimensions_xz[0])
    depth_z = float(dimensions_xz[1])
    
    # Compute the 4 corners of the OBB in XZ for overlap checks
    corners_local = np.array([
        [min_xz[0], min_xz[1]],
        [max_xz[0], min_xz[1]],
        [max_xz[0], max_xz[1]],
        [min_xz[0], max_xz[1]]
    ])
    corners_global = (corners_local @ evecs.T) + xz_centroid
    
    # Construct 3x3 Rotation matrix (Y is unchanged)
    rot_matrix = np.eye(3)
    rot_matrix[0, 0] = evecs[0, 0]
    rot_matrix[0, 2] = evecs[0, 1]
    rot_matrix[2, 0] = evecs[1, 0]
    rot_matrix[2, 2] = evecs[1, 1]
    
    return {
        "centroid": {"x": float(centroid[0]), "y": float(centroid[1]), "z": float(centroid[2])},
        "dimensions": {"width_x": width_x, "height_y": height_y, "depth_z": depth_z},
        "rotation_matrix": [list(row) for row in rot_matrix],
        "bounds": {"min_y": min_y, "max_y": max_y},
        "xz_corners": [list(c) for c in corners_global]
    }

def check_xz_polygon_overlap(corners_a, corners_b):
    """Use Separating Axis Theorem (SAT) to check if two convex polygons overlap."""
    polygons = [np.array(corners_a), np.array(corners_b)]
    for polygon in polygons:
        for i in range(4):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % 4]
            normal = np.array([p2[1] - p1[1], p1[0] - p2[0]]) # Perpendicular to edge
            
            # Project all points of both polygons onto the normal
            min_a, max_a = float('inf'), float('-inf')
            for p in corners_a:
                proj = np.dot(p, normal)
                min_a, max_a = min(min_a, proj), max(max_a, proj)
                
            min_b, max_b = float('inf'), float('-inf')
            for p in corners_b:
                proj = np.dot(p, normal)
                min_b, max_b = min(min_b, proj), max(max_b, proj)
                
            # If there's a gap along this normal, they do not overlap
            if max_a < min_b or max_b < min_a:
                return False
    return True

def generate_scene_graph(config_path: str = "/workspace/configs/pipeline_config.json"):
    data_root = Path("/workspace/data")
    meta_path = data_root / "outputs" / "masks" / "segmentation_metadata.json"
    
    if not meta_path.exists():
        print(f"Error: Metadata not found at {meta_path}. Run segmentation first.")
        return

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    splat_path = Path(meta['splat_path'])
    if not splat_path.exists():
        splat_path = data_root / "outputs" / "splats" / "sugar_output" / "sugar_refined.ply"

    print(f"Loading 3D points from {splat_path}...")
    plydata = PlyData.read(str(splat_path))
    vertices = plydata['vertex']
    
    xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    scene_graph = {
        "objects": {},
        "relationships": []
    }
    
    objects = meta.get('objects', [])
    print(f"Extracting geometric OBBs for {len(objects)} objects...")
    
    for obj_name in objects:
        indices = meta['gaussian_indices'].get(obj_name, [])
        if not indices:
            continue
            
        pts = xyz[indices]
        clean_pts = filter_outliers(pts, percentile=5)
        if len(clean_pts) < 10:
            clean_pts = pts 
            
        obb_data = compute_xz_obb(clean_pts)
        obb_data["point_count"] = len(indices)
        
        scene_graph["objects"][obj_name] = obb_data
        
    print("Computing pairwise spatial relationships...")
    obj_names = list(scene_graph["objects"].keys())
    
    for i in range(len(obj_names)):
        for j in range(len(obj_names)):
            if i == j: continue
            
            name_a = obj_names[i]
            name_b = obj_names[j]
            
            obj_a = scene_graph["objects"][name_a]
            obj_b = scene_graph["objects"][name_b]
            
            ca = np.array([obj_a["centroid"]["x"], obj_a["centroid"]["y"], obj_a["centroid"]["z"]])
            cb = np.array([obj_b["centroid"]["x"], obj_b["centroid"]["y"], obj_b["centroid"]["z"]])
            
            # Metric Distance
            distance = float(np.linalg.norm(ca - cb))
            
            rel = {
                "subject": name_a,
                "object": name_b,
                "distance_meters": distance,
                "predicates": []
            }
            
            # Vertical relationship (Y is up in ARKit)
            is_above = obj_a['bounds']['min_y'] > obj_b['bounds']['max_y']
            is_below = obj_a['bounds']['max_y'] < obj_b['bounds']['min_y']
            xz_overlap = check_xz_polygon_overlap(obj_a['xz_corners'], obj_b['xz_corners'])
            
            if xz_overlap:
                if is_above:
                    rel["predicates"].append("directly_above")
                    # If it's directly above and very close
                    vertical_gap = obj_a['bounds']['min_y'] - obj_b['bounds']['max_y']
                    if vertical_gap < 0.2: # within 20 cm
                        rel["predicates"].append("resting_on")
                elif is_below:
                    rel["predicates"].append("directly_below")
            
            # Horizontal proximity
            if distance < 1.0: # 1 meter criteria for next to
                rel["predicates"].append("next_to")
            else:
                if distance >= 2.0:
                    rel["predicates"].append("across_room")
                
                # Determine primary relative direction mapped to axes
                dx = cb[0] - ca[0]
                dz = cb[2] - ca[2]
                if abs(dx) > abs(dz):
                    if dx > 0:
                        rel["predicates"].append("towards_positive_x")
                    else:
                        rel["predicates"].append("towards_negative_x")
                else:
                    if dz > 0:
                        rel["predicates"].append("towards_positive_z")
                    else:
                        rel["predicates"].append("towards_negative_z")
                
            scene_graph["relationships"].append(rel)

    output_path = data_root / "outputs" / "masks" / "scene_graph.json"
    print(f"Saving Scene Graph to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(scene_graph, f, indent=4)
        
    print("🎉 Scene graph generation complete! It is now ready for querying.")

if __name__ == "__main__":
    generate_scene_graph()
