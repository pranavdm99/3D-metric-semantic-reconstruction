import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from align_trajectory import parse_camtrackar

def main():
    parser = argparse.ArgumentParser(description="Plot 3D Trajectory from AR data.")
    parser.add_argument('--ar_data', type=str, required=True, help="Directory containing the AR tracking data (.hfcs)")
    parser.add_argument('--colmap_images', type=str, default=None, help="Path to COLMAP images.txt for overlay")
    parser.add_argument('--output', type=str, required=True, help="Output path for the trajectory plot (.png)")
    args = parser.parse_args()

    ar_data_dir = Path(args.ar_data)
    hfcs_files = list(ar_data_dir.glob('*.hfcs'))
    
    if not hfcs_files:
        print(f"Error: Could not find any .hfcs file in {ar_data_dir}")
        sys.exit(1)
        
    hfcs_path = hfcs_files[0]
    print(f"Loading trajectory from {hfcs_path}")
    
    positions, fps = parse_camtrackar(str(hfcs_path))
    
    # Parse COLMAP trajectory if provided
    colmap_positions = None
    if args.colmap_images and os.path.exists(args.colmap_images):
        print(f"Loading SfM trajectory from {args.colmap_images}")
        from scipy.spatial.transform import Rotation
        colmap_pos_list = []
        with open(args.colmap_images, 'r') as f:
            lines = f.readlines()
        
        # In COLMAP images.txt, camera lines alternate with 2D point lines. Camera lines have 10+ elements.
        for line in lines:
            if line.startswith('#'): continue
            parts = line.strip().split()
            # Odd lines have image info: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            if len(parts) >= 10 and parts[0].isdigit() and parts[8].isdigit():
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                # scipy expects x, y, z, w
                r = Rotation.from_quat([qx, qy, qz, qw])
                t = np.array([tx, ty, tz])
                camera_center = -r.as_matrix().T @ t
                colmap_pos_list.append(camera_center)
                
        if colmap_pos_list:
            colmap_positions = np.array(colmap_pos_list)
            print(f"Loaded {len(colmap_positions)} SfM poses.")

    fig = plt.figure(figsize=(16, 8))
    
    # --- 3D Trajectory Plot ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='ARKit Trajectory', color='b', marker='.', markersize=2, linewidth=1)
    
    if colmap_positions is not None:
        ax1.plot(colmap_positions[:, 0], colmap_positions[:, 1], colmap_positions[:, 2], 
                 label='COLMAP SfM Path', color='orange', marker='+', markersize=4, linestyle='dashed')

    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='g', s=100, label='AR Start', zorder=5)
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='r', s=100, label='AR End', zorder=5)

    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_zlabel('Z (meters)')
    ax1.set_title('3D Camera Trajectory Overlay')
    ax1.legend()
    
    # Make 3D axes equal for realistic viewing
    all_pos = positions
    if colmap_positions is not None:
        all_pos = np.vstack((positions, colmap_positions))
        
    max_range = np.array([all_pos[:,0].max()-all_pos[:,0].min(), 
                          all_pos[:,1].max()-all_pos[:,1].min(), 
                          all_pos[:,2].max()-all_pos[:,2].min()]).max() / 2.0
    mid_x = (all_pos[:,0].max()+all_pos[:,0].min()) * 0.5
    mid_y = (all_pos[:,1].max()+all_pos[:,1].min()) * 0.5
    mid_z = (all_pos[:,2].max()+all_pos[:,2].min()) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # --- 2D Top-Down (Bird's Eye) Plot (XZ Plane) ---
    ax2 = fig.add_subplot(122)
    ax2.plot(positions[:, 0], positions[:, 2], label='ARKit Path (XZ)', color='b', marker='.', markersize=2, linewidth=1)
    
    if colmap_positions is not None:
        ax2.plot(colmap_positions[:, 0], colmap_positions[:, 2], label='SfM Path (XZ)', color='orange', marker='+', markersize=4, linestyle='dashed')

    ax2.scatter(positions[0, 0], positions[0, 2], color='g', s=100, label='AR Start', zorder=5)
    ax2.scatter(positions[-1, 0], positions[-1, 2], color='r', s=100, label='AR End', zorder=5)

    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Z (meters)')
    ax2.set_title("Bird's Eye View (Top-Down)")
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')  # Ensure X and Z scales are identical

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved trajectory plot to {args.output}")

if __name__ == '__main__':
    main()
