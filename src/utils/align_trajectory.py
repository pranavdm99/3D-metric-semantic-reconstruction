import os
import sys
import json
import xml.etree.ElementTree as ET
import glob
from pathlib import Path
import numpy as np
import argparse

def parse_beeble(json_path):
    print(f"Parsing Beeble AR data from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    video_fps = data.get('frame_rate', 30)
    raw_data = data.get('raw_data', [])
    
    positions = []
    for frame in raw_data:
        mat = np.array(frame['cameraData']['transform']).reshape(4, 4)
        # Assuming transform is Camera-to-World in meters
        pos = mat[:3, 3]
        positions.append(pos)
        
    return np.array(positions), video_fps

def parse_camtrackar(hfcs_path):
    print(f"Parsing CamTrackAR data from {hfcs_path}")
    tree = ET.parse(hfcs_path)
    root = tree.getroot()
    
    avSettings = root.find(".//*AudioVideoSettings")
    video_fps = int(avSettings.find("FrameRate").text)
    
    cameraNode = root.find(".//*CameraLayer")
    posAnim = cameraNode.find(".//*position/Animation")
    
    # Timeline is in ms since <TimelineTimeFormat>1000</TimelineTimeFormat>
    times = []
    positions = []
    
    for key in posAnim.findall('Key'):
        times.append(float(key.get('Time')) / 1000.0)
        p = key.find('.//*FXPoint3_32f')
        positions.append([float(p.get('X')), float(p.get('Y')), float(p.get('Z'))])
        
    positions = np.array(positions)
    # CamTrackAR HFCS units are scaled. Blender script uses: 1/1000 * 2.8352
    positions = positions * 0.0028352
    times = np.array(times)
    
    # We want to resample them to a constant video_fps to match Beeble's interface intuitively
    # Or we can just build an interpolator
    from scipy.interpolate import interp1d
    interp_pos = interp1d(times, positions, axis=0, bounds_error=False, fill_value="extrapolate")
    
    # Generate array for each frame index
    max_time = times[-1]
    num_frames = int(max_time * video_fps) + 1
    query_times = np.arange(num_frames) / video_fps
    
    resampled_positions = interp_pos(query_times)
    return resampled_positions, video_fps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ar_data', type=str, required=True, help="Directory containing AR tracking data")
    parser.add_argument('--fps', type=float, default=2.0, help="Extraction FPS used to create frames")
    parser.add_argument('--output', type=str, required=True, help="Output ref_images.txt path for COLMAP")
    args = parser.parse_args()

    ar_data_dir = Path(args.ar_data)
    beeble_json = ar_data_dir / 'camera.json'
    camtrackar_hfcs = list(ar_data_dir.glob('*.hfcs'))
    
    positions = None
    video_fps = None
    
    if beeble_json.exists():
        positions, video_fps = parse_beeble(beeble_json)
    elif camtrackar_hfcs:
        positions, video_fps = parse_camtrackar(camtrackar_hfcs[0])
    else:
        print("Error: Could not find supported AR data (Beeble camera.json or CamTrackAR .hfcs) in", ar_data_dir)
        sys.exit(1)
        
    print(f"Loaded {len(positions)} AR frames at {video_fps} FPS.")
    
    # Map extracted frames to AR positions
    # Video extraction via `ffmpeg -vf fps=X` gives frame_0001 at roughly t=0.0
    # frame_i corresponds to t = (i-1) / fps
    total_duration = len(positions) / video_fps
    max_extracted_frames = int(total_duration * args.fps) + 5
    
    with open(args.output, 'w') as f:
        for i in range(1, max_extracted_frames + 1):
            t = (i - 1) / float(args.fps)
            frame_idx = int(round(t * video_fps))
            
            if frame_idx < len(positions):
                pos = positions[frame_idx]
                image_name = f"frame_{i:04d}.jpg"
                f.write(f"{image_name} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
    
    print(f"Saved COLMAP reference positions to {args.output}")

if __name__ == "__main__":
    main()
