#!/usr/bin/env python3
"""
Segmentation: Semantic Grounding & 3D Segmentation
Uses Qwen3-VL for object detection and SAM2 for mask tracking
"""

import os
import sys
import json
import argparse
import subprocess
import gc
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import cv2
from tqdm import tqdm
import requests
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLOWorld


class ColmapParser:
    """Parser for COLMAP text-format output (cameras.txt, images.txt)"""
    
    @staticmethod
    def read_cameras(cameras_path: Path) -> Dict[int, Dict]:
        cameras = {}
        with open(cameras_path, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip(): continue
                parts = line.split()
                cam_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]
                
                # SuGaR uses PINHOLE/SIMPLE_PINHOLE
                if model == "SIMPLE_PINHOLE":
                    f_val, cx, cy = params
                    k = np.array([[f_val, 0, cx], [0, f_val, cy], [0, 0, 1]])
                elif model == "PINHOLE":
                    fx, fy, cx, cy = params
                    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                else:
                    k = np.eye(3) # Fallback
                
                cameras[cam_id] = {"width": width, "height": height, "K": k}
        return cameras

    @staticmethod
    def read_images(images_path: Path) -> Dict[str, Dict]:
        images = {}
        with open(images_path, "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                line = lines[i]
                if line.startswith("#") or not line.strip(): continue
                parts = line.split()
                # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                qw, qx, qy, qz = [float(p) for p in parts[1:5]]
                tx, ty, tz = [float(p) for p in parts[5:8]]
                cam_id = int(parts[8])
                name = parts[9]
                
                rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
                t = np.array([tx, ty, tz])
                
                images[name] = {"R": rot, "t": t, "cam_id": cam_id}
        return images


class YOLOWorldDetector:
    """Zero-shot object detector using YOLO World"""
    
    def __init__(self, model_id: str = "yolov8s-worldv2.pt"):
        # Fix for PyTorch 2.6+ weights_only=True security restriction
        # Adding specifically requested classes to the safe list
        try:
            import torch.serialization
            import torch.nn
            from ultralytics.nn.tasks import WorldModel
            if hasattr(torch.serialization, 'add_safe_globals'):
                # Essential standard and ultralytics classes
                safe_classes = [
                    torch.nn.modules.container.Sequential,
                    torch.nn.modules.container.ModuleList,
                    torch.nn.functional.silu,
                    WorldModel
                ]
                # Add more common ultralytics modules to avoid whack-a-mole
                try:
                    from ultralytics.nn.modules.conv import Conv, Concat
                    from ultralytics.nn.modules.block import C2f, Bottleneck, SPPELAN
                    safe_classes.extend([Conv, Concat, C2f, Bottleneck, SPPELAN])
                except ImportError:
                    pass
                torch.serialization.add_safe_globals(safe_classes)
        except Exception:
            pass

        print(f"🚀 Initializing YOLO World with {model_id}...")
        self.model = YOLOWorld(model_id)
        # Explicitly move to device to avoid DeviceMismatch later
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
    def detect_objects(self, image_path: Path, object_queries: List[str]) -> List[Dict]:
        """Detect objects using YOLO World"""
        # Set classes
        self.model.set_classes(object_queries)
        
        # Run inference - explicitly set device
        results = self.model.predict(str(image_path), conf=0.1, verbose=False, device=self.device)
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            names = result.names
            
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                coords = box.xyxyn[0].cpu().numpy().tolist() # [x1, y1, x2, y2] normalized 0-1
                conf = float(box.conf[0])
                
                detections.append({
                    "object": label,
                    "bbox": coords,
                    "confidence": conf
                })
        
        return detections

    def unload_model(self):
        """Free memory"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()


class SAM2Segmenter:
    """Wrapper for SAM 2 video segmentation"""
    
    def __init__(self, checkpoint_path: Path):
        from sam2.build_sam import build_sam2_video_predictor
        
        self.checkpoint = checkpoint_path
        self.predictor = build_sam2_video_predictor(
            "sam2_hiera_s.yaml",
            str(checkpoint_path)
        )
        
    def segment_and_track(
        self, 
        video_dir: Path,
        prompts: List[Tuple[str, List[float], int]],
        output_dir: Path,
        batch_size: int = 4
    ) -> Dict[str, np.ndarray]:
        """
        Segment objects and track across frames using multi-indexed prompts.
        Processed in batches to manage VRAM.
        """
        print(f"🎭 Running SAM 2 segmentation and tracking (Batch Size: {batch_size})...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frames = sorted(video_dir.glob('*.jpg'))
        if not frames:
            frames = sorted(video_dir.glob('*.png'))
        img = cv2.imread(str(frames[0]))
        h, w = img.shape[:2]
        
        # SAM 2 strictly expects numeric filenames in some environments
        tmp_frames_dir = Path("/tmp/sam2_frames")
        if not tmp_frames_dir.exists():
            tmp_frames_dir.mkdir(parents=True)
            for i, frame_path in enumerate(frames):
                (tmp_frames_dir / f"{i:05d}.jpg").symlink_to(frame_path)
        
        all_object_masks = {}
        
        # Process in batches
        for batch_idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_idx : batch_idx + batch_size]
            print(f"\n📦 Processing batch {batch_idx//batch_size + 1}/{(len(prompts)-1)//batch_size + 1} ({len(batch_prompts)} objects)...")
            
            inference_state = self.predictor.init_state(video_path=str(tmp_frames_dir))
            batch_object_names = []
            
            for i, (obj_class, bbox, f_idx) in enumerate(batch_prompts):
                obj_id = i
                obj_name = f"{obj_class}_{batch_idx + i}"
                batch_object_names.append(obj_name)
                
                # Convert normalized bbox to pixel coordinates
                x_min, y_min, x_max, y_max = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
                box = np.array([x_min, y_min, x_max, y_max])
                
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=f_idx,
                    obj_id=obj_id,
                    box=box,
                )
            
            batch_masks = {name: [None] * len(frames) for name in batch_object_names}
            prompt_frames = [p[2] for p in batch_prompts]
            min_p, max_p = min(prompt_frames), max(prompt_frames)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward
                for out_f, out_ids, out_logits in self.predictor.propagate_in_video(inference_state, min_p, reverse=False):
                    for i, out_id in enumerate(out_ids):
                        mask = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                        batch_masks[batch_object_names[out_id]][out_f] = mask
                # Backward
                for out_f, out_ids, out_logits in self.predictor.propagate_in_video(inference_state, max_p, reverse=True):
                    for i, out_id in enumerate(out_ids):
                        mask = (out_logits[i] > 0.0).cpu().numpy().squeeze()
                        batch_masks[batch_object_names[out_id]][out_f] = mask
            
            # Save batch results and clear memory
            for name, masks in batch_masks.items():
                obj_dir = output_dir / name
                obj_dir.mkdir(exist_ok=True)
                for i in range(len(frames)):
                    m = masks[i] if masks[i] is not None else np.zeros((h, w), dtype=bool)
                    cv2.imwrite(str(obj_dir / f"mask_{i:04d}.png"), (m * 255).astype(np.uint8))
                all_object_masks[name] = masks
                print(f"  ✅ Saved masks for {name}")
            
            # CRITICAL: Clean up memory before next batch
            self.predictor.reset_state(inference_state)
            del inference_state
            gc.collect()
            torch.cuda.empty_cache()
            
        return all_object_masks


class SegmentationPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)['segmentation']
        
        self.data_root = Path(self.config['data_root'])
        self.frames_dir = self.data_root / 'raw_video' / 'frames'
        self.output_dir = self.data_root / 'outputs' / 'masks'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Reconstruction output
        with open(self.data_root / 'outputs' / 'splats' / 'reconstruction_metadata.json') as f:
            self.reconstruction_meta = json.load(f)
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate intersection over union for two normalized bboxes [x1, y1, x2, y2]"""
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        if box1Area + box2Area - interArea <= 0: return 0.0
        return interArea / (box1Area + box2Area - interArea)

    def select_keyframe(self, fraction: float = 0.5) -> Tuple[Path, int]:
        """Select a representative keyframe and return path + index"""
        frames = sorted(self.frames_dir.glob('*.jpg'))
        index = int(len(frames) * fraction)
        keyframe = frames[index]
        print(f"🖼️  Selected keyframe: {keyframe.name} (at {fraction*100:.0f}% of video, index {index})")
        return keyframe, index
    
    def lift_masks_to_3d(self, object_masks: Dict[str, np.ndarray], splat_path: Path):
        """Map 2D masks back to 3D Gaussians using camera projection voting"""
        print("🎯 Lifting 2D masks to 3D Gaussian groups using camera projection...")
        
        from plyfile import PlyData, PlyElement
        
        # 1. Load COLMAP camera data
        colmap_root = self.data_root / 'outputs' / 'colmap' / 'sparse' / '0'
        if not (colmap_root / 'cameras.txt').exists():
            print("  ⚠️  COLMAP data not found. Falling back to axis partition.")
            return self._lift_masks_fallback(object_masks, splat_path)
            
        cameras = ColmapParser.read_cameras(colmap_root / 'cameras.txt')
        image_metas = ColmapParser.read_images(colmap_root / 'images.txt')
        
        # 2. Load splat
        plydata = PlyData.read(str(splat_path))
        vertices = plydata['vertex']
        xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        n_gaussians = len(xyz)
        
        # 3. Initialize voting matrix [n_gaussians, n_objects]
        object_names = list(object_masks.keys())
        if not object_names:
             print("  ⚠️  No objects detected to lift.")
             return {}
             
        n_objects = len(object_names)
        votes = np.zeros((n_gaussians, n_objects), dtype=np.float32)
        
        # 4. Multi-view voting
        sample_rate = 5 # Speed up by sampling frames
        n_frames = len(list(object_masks.values())[0]) if object_masks else 0
        masked_frames = list(range(0, n_frames, sample_rate))
        
        print(f"  → Voting across {len(masked_frames)} keyframes...")
        all_frames = sorted(self.frames_dir.glob('*.jpg'))
        
        # Load one frame to get resolution ratios
        if all_frames:
            ref_img = cv2.imread(str(all_frames[0]))
            img_h, img_w = ref_img.shape[:2]
        else:
            img_h, img_w = 1080, 1920 # Fallback
            
        for f_idx in masked_frames:
            if f_idx >= len(all_frames): break
            frame_name = all_frames[f_idx].name
            if frame_name not in image_metas: continue
            
            meta = image_metas[frame_name]
            cam = cameras[meta['cam_id']]
            K, R_mat, t = cam['K'].copy(), meta['R'], meta['t']
            
            # CRITICAL: Scale intrinsic K if the mask resolution differs from COLMAP resolution
            if cam['width'] != img_w or cam['height'] != img_h:
                scale_x = img_w / cam['width']
                scale_y = img_h / cam['height']
                K[0, 0] *= scale_x
                K[0, 2] *= scale_x
                K[1, 1] *= scale_y
                K[1, 2] *= scale_y

            # CRITICAL: Fix for COLMAP rotation convention and world-to-cam projection
            # COLMAP stores R and t for x_c = R * x_w + t
            xyz_cam = (R_mat @ xyz.T).T + t
            
            # Diagnostic for Frame 0 only
            if f_idx == masked_frames[0]:
                print(f"    🔍 Diagnostic (Frame {f_idx}):")
                print(f"       - K matrix: \n{K}")
                print(f"       - Splat XYZ range: {np.min(xyz, axis=0)} to {np.max(xyz, axis=0)}")
                print(f"       - Depth range in camera: {np.min(xyz_cam[:, 2]):.2f} to {np.max(xyz_cam[:, 2]):.2f}")
            
            # 2D projection
            mask_depth = xyz_cam[:, 2] > 0.1
            valid_depth_idx = np.where(mask_depth)[0]
            if len(valid_depth_idx) == 0: continue
            
            uv_homo = (K @ xyz_cam[valid_depth_idx].T).T
            u = (uv_homo[:, 0] / uv_homo[:, 2]).astype(int)
            v = (uv_homo[:, 1] / uv_homo[:, 2]).astype(int)
            
            # Clip to image bounds
            mask_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
            in_bounds_idx = valid_depth_idx[mask_bounds]
            u_valid = u[mask_bounds]
            v_valid = v[mask_bounds]
            
            if f_idx == masked_frames[0]:
                print(f"       - Points in bounds: {len(in_bounds_idx):,}")
            
            if len(in_bounds_idx) == 0: continue
            
            for obj_idx, obj_name in enumerate(object_names):
                obj_mask = object_masks[obj_name][f_idx]
                obj_votes = obj_mask[v_valid, u_valid]
                votes[in_bounds_idx[obj_votes], obj_idx] += 1.0
                
        # 5. Assign objects based on maximum votes
        # Lower threshold: require objects to be seen in at least 5% of keyframes
        min_votes = max(1, len(masked_frames) // 20)
        labels = np.argmax(votes, axis=1)
        max_votes = np.max(votes, axis=1)
        
        # Initial set of indices
        obj_indices = {}
        for obj_idx, obj_name in enumerate(object_names):
            match_idx = np.where((labels == obj_idx) & (max_votes >= min_votes))[0]
            if len(match_idx) >= 50: # Minimum size pruning
                obj_indices[obj_name] = set(match_idx.tolist())
        
        # 6. 3D Merging (Set intersection over 50%)
        print(f"  → Merging 3D Gaussian groups using set intersection...")
        final_gaussian_indices = {}
        processed_names = set()
        
        sorted_names = sorted(obj_indices.keys(), key=lambda n: len(obj_indices[n]), reverse=True)
        
        for name_a in sorted_names:
            if name_a in processed_names: continue
            set_a = obj_indices[name_a]
            
            # This is the master set for this resulting object
            master_set = set_a
            merged_names = [name_a]
            processed_names.add(name_a)
            
            for name_b in sorted_names:
                if name_b in processed_names: continue
                set_b = obj_indices[name_b]
                
                # Intersection over Union for 3D sets
                intersection = master_set.intersection(set_b)
                union = master_set.union(set_b)
                iou_3d = len(intersection) / len(union) if len(union) > 0 else 0
                
                # Also check if B is largely contained within A
                overlap_ratio = len(intersection) / len(set_b) if len(set_b) > 0 else 0
                
                if iou_3d > 0.5 or overlap_ratio > 0.8:
                    print(f"     🔗 Merging '{name_b}' into '{name_a}' (3D IoU: {iou_3d:.2f}, Overlap: {overlap_ratio:.2f})")
                    master_set = union
                    merged_names.append(name_b)
                    processed_names.add(name_b)
            
            # Save the largest one or a combined name? User asked to remove duplicates.
            # We'll stick with the name_a as it was the largest/first.
            final_gaussian_indices[name_a] = list(master_set)

        print(f"  → Final Voting Diagnostics (min_votes={min_votes}, min_points=50):")
        for obj_name, idxs in final_gaussian_indices.items():
            print(f"     - {obj_name}: {len(idxs):,} assigned.")

        # 7. Save metadata
        mask_metadata = {
            'objects': list(final_gaussian_indices.keys()),
            'n_frames': len(masked_frames),
            'splat_path': str(splat_path),
            'n_gaussians': n_gaussians,
            'gaussian_indices': {k: list(v) for k,v in final_gaussian_indices.items()},
            'lifting_method': 'camera_projection_voting_with_3d_merging'
        }
        
        with open(self.output_dir / 'segmentation_metadata.json', 'w') as f:
            json.dump(mask_metadata, f, indent=2)
        
        print(f"  ✅ Mask metadata saved")
        return mask_metadata

    def _lift_masks_fallback(self, object_masks: Dict[str, np.ndarray], splat_path: Path):
        """Original axis-partition fallback logic"""
        from plyfile import PlyData
        plydata = PlyData.read(str(splat_path))
        vertices = plydata['vertex']
        n_gaussians = len(vertices)
        object_names = list(object_masks.keys())
        
        mask_areas = []
        for obj_name in object_names:
            frames = object_masks.get(obj_name, [])
            if not frames:
                mask_areas.append(1e-6)
                continue
            areas = [float(np.mean(mask.astype(np.float32))) for mask in frames]
            mask_areas.append(max(1e-6, float(np.mean(areas))))

        weights = np.array(mask_areas, dtype=np.float64)
        weights = weights / (np.sum(weights) + 1e-9)

        xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        axis_spread = np.ptp(xyz, axis=0)
        dominant_axis = int(np.argmax(axis_spread))
        sorted_indices = np.argsort(xyz[:, dominant_axis])

        counts = np.floor(weights * n_gaussians).astype(int)
        gaussian_indices: Dict[str, List[int]] = {}
        start = 0
        for i, obj_name in enumerate(object_names):
            end = start + int(counts[i])
            gaussian_indices[obj_name] = sorted_indices[start:end].astype(np.int64).tolist()
        return {'gaussian_indices': gaussian_indices, 'lifting_method': 'axis_partition_fallback'}
    
    def run(self):
        """Execute the full Segmentation pipeline"""
        print("=" * 60)
        print("SEGMENTATION: Semantic Grounding & 3D Segmentation")
        print("=" * 60)
        
        # Get target objects from config
        target_objects = self.config.get('target_objects', ['chair', 'table', 'mug'])
        print(f"🎯 Target objects: {target_objects}")
        
        # Step 1: YOLO World Multi-Anchor Discovery
        print("\n📍 Step 1: Multi-anchor object discovery")
        all_prompts = [] # List of (class, bbox, frame_idx)
        try:
            detector = YOLOWorldDetector()
            # Increase sampling frequency to 10 anchors
            anchor_fractions = np.linspace(0.05, 0.95, 10)
            
            # Step 1.1: Multi-anchor discovery loop
            from collections import defaultdict
            class_detections_by_frame = defaultdict(list)
            
            for fraction in anchor_fractions:
                keyframe_path, frame_idx = self.select_keyframe(fraction)
                detections = detector.detect_objects(keyframe_path, target_objects)
                for det in detections:
                    # Store with frame_idx for deduplication
                    det['frame_idx'] = frame_idx
                    class_detections_by_frame[frame_idx].append(det)
            
            detector.unload_model()
            torch.cuda.empty_cache()
            
            # Step 1.2: Deduplicate class_detections per frame (IoU > 0.8)
            print(f"  → Deduplicating overlapping detections (2D IoU > 0.8)...")
            final_prompt_detections = []
            for f_idx in sorted(class_detections_by_frame.keys()):
                frame_dets = class_detections_by_frame[f_idx]
                keep = []
                for det_a in frame_dets:
                    is_duplicate = False
                    for det_b in keep:
                        if self._calculate_iou(det_a['bbox'], det_b['bbox']) > 0.8:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        keep.append(det_a)
                class_detections_by_frame[f_idx] = keep

            # Step 1.3: For each class, identify unique instances to prompt
            all_prompts = []
            for obj_class in target_objects:
                # Find max instances of this class seen simultaneously in ANY anchor frame
                max_inst = 0
                for f_idx, dets in class_detections_by_frame.items():
                    c_count = sum(1 for d in dets if d['object'] == obj_class)
                    max_inst = max(max_inst, c_count)
                
                if max_inst == 0:
                    print(f"  ⚠️  '{obj_class}' not detected in any anchor frame.")
                    continue
                    
                print(f"  → Found up to {max_inst} instances of '{obj_class}'")
                for i in range(max_inst):
                    # Find earliest frame that has at least i+1 of this class
                    found = False
                    for f_idx in sorted(class_detections_by_frame.keys()):
                        insts = [d for d in class_detections_by_frame[f_idx] if d['object'] == obj_class]
                        if len(insts) > i:
                            all_prompts.append((obj_class, insts[i]['bbox'], f_idx))
                            found = True
                            break
            
            if not all_prompts:
                print("❌ ERROR: No objects detected in any anchor! Check input video.")
                print("  → Pipeline will attempt fallback but results will be poor.")
            
            # Step 2: SAM 2 temporal masking
            print("\n🎭 Step 2: SAM 2 temporal segmentation")
            sam_checkpoint = Path('/workspace/checkpoints/sam2_hiera_small.pt')
            segmenter = SAM2Segmenter(sam_checkpoint)
            
            object_masks = segmenter.segment_and_track(
                self.frames_dir,
                all_prompts,
                self.output_dir
            )
                    
        except (RuntimeError, Exception) as err:
            print(f"  ⚠️  Segmentation Detection Pipeline Error: {err}")
            print("  → Using fallback: empty object assignment")
            object_masks = {obj: [] for obj in target_objects}
            for obj_dir in [self.output_dir / obj for obj in target_objects]:
                obj_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear VRAM (only if segmenter was created)
        if 'segmenter' in locals():
            del segmenter
        torch.cuda.empty_cache()
        
        # Step 3: 3D lifting
        print("\n🚀 Step 3: Lift masks to 3D")
        splat_path = Path(self.reconstruction_meta['ply_path'])
        mask_metadata = self.lift_masks_to_3d(object_masks, splat_path)
        
        print("\n" + "=" * 60)
        print("✅ SEGMENTATION COMPLETE!")
        print("=" * 60)
        print(f"📦 Masks saved to: {self.output_dir}")
        print(f"📊 Metadata: {self.output_dir / 'segmentation_metadata.json'}")
        print(f"🎯 Detected objects: {len(object_masks)}")
        print("\n➡️  Next: Run physics extraction for material estimation")


def main():
    parser = argparse.ArgumentParser(description='Segmentation: Semantic Segmentation')
    parser.add_argument('--config', type=str, default='/workspace/configs/pipeline_config.json',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    pipeline = SegmentationPipeline(args.config)
    pipeline.run()


if __name__ == '__main__':
    main()
