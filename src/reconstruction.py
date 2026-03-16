#!/usr/bin/env python3
"""
Reconstruction: Neural Reconstruction & Surface Alignment
Converts multi-view video into surface-aligned 3D Gaussian Splats using SuGaR
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import torch
import numpy as np
from plyfile import PlyData, PlyElement


class ReconstructionPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)['reconstruction']
        
        self.data_root = Path(self.config['data_root'])
        self.video_path = Path(self.config['input_video'])
        self.output_dir = self.data_root / 'outputs' / 'splats'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # VRAM optimization
        self.max_vram_gb = 7.5  # Leave 0.5GB headroom
        
    def extract_frames(self):
        """Extract frames from multi-view video"""
        print("Extracting frames from video...")
        frames_dir = self.data_root / 'raw_video' / 'frames'
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract at configurable FPS to reduce memory footprint
        fps = self.config.get('extraction_fps', 2)
        cmd = [
            'ffmpeg', '-i', str(self.video_path),
            '-vf', f'fps={fps}',
            '-qscale:v', '2',
            str(frames_dir / 'frame_%04d.jpg')
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Extracted frames to {frames_dir}")
        return frames_dir
    
    def run_colmap(self, frames_dir: Path):
        """Run COLMAP Structure from Motion"""
        print("Running COLMAP for camera pose estimation...")
        colmap_dir = self.data_root / 'outputs' / 'colmap'
        colmap_dir.mkdir(parents=True, exist_ok=True)
        
        # SuGaR expects images in a specific location
        images_dir = colmap_dir / 'images'
        if images_dir.exists():
            if images_dir.is_symlink():
                images_dir.unlink()
            else:
                import shutil
                shutil.rmtree(images_dir)
        
        try:
            # Use absolute path instead of symlink if possible
            subprocess.run(['ln', '-s', str(frames_dir), str(images_dir)], check=True)
        except Exception:
            import shutil
            shutil.copytree(frames_dir, images_dir)
        
        colmap_env = os.environ.copy()
        colmap_env.setdefault('QT_QPA_PLATFORM', 'offscreen')
        
        database_path = colmap_dir / 'database.db'
        if database_path.exists():
            print(f"Removing stale database: {database_path}")
            database_path.unlink()

        sparse_dir = colmap_dir / 'sparse'
        if sparse_dir.exists():
            print(f"Cleaning stale sparse models in: {sparse_dir}")
            import shutil
            shutil.rmtree(sparse_dir)
        sparse_dir.mkdir(exist_ok=True)
        
        # Feature extraction (memory-efficient settings)
        print("Feature extraction...")
        subprocess.run([
            'colmap', 'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', str(frames_dir),
            '--ImageReader.camera_model', 'SIMPLE_PINHOLE',  # SuGaR requires PINHOLE or SIMPLE_PINHOLE
            '--SiftExtraction.max_image_size', '1600',
            '--SiftExtraction.max_num_features', '4096',
            '--SiftExtraction.use_gpu', '0'
        ], check=True, env=colmap_env)
        
        # Feature matching
        print("Feature matching...")
        subprocess.run([
            'colmap', 'exhaustive_matcher',
            '--database_path', str(database_path),
            '--SiftMatching.guided_matching', '1',
            '--SiftMatching.use_gpu', '0'
        ], check=True, env=colmap_env)
        
        # Bundle adjustment
        print("Bundle adjustment...")
        subprocess.run([
            'colmap', 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(frames_dir),
            '--output_path', str(sparse_dir),
            '--Mapper.ba_global_max_num_iterations', '20'  # Reduce iterations
        ], check=True, env=colmap_env)
        
        # Verify reconstruction success
        if not (sparse_dir / '0').exists():
            print("COLMAP Error: No sparse model (sparse/0) was created.")
            print("This usually means SfM failed to find enough matches or a good initial pair.")
            print("Check your input video quality and coverage.")
            raise RuntimeError("COLMAP failed to create a reconstruction model.")

        # Convert to text format for easier processing
        subprocess.run([
            'colmap', 'model_converter',
            '--input_path', str(sparse_dir / '0'),
            '--output_path', str(sparse_dir / '0'),
            '--output_type', 'TXT'
        ], check=True, env=colmap_env)
        
        print(f"COLMAP output saved to {colmap_dir}")
        return colmap_dir
    
    def patch_sugar(self):
        """Patch SuGaR source code to expose hardcoded parameters"""
        print("Patching SuGaR for memory efficiency...")
        
        sugar_root = Path("/workspace/SuGaR")
        trainer_path = sugar_root / "sugar_trainers" / "coarse_density.py"
        wrapper_path = sugar_root / "train_coarse_density.py"
        cameras_path = sugar_root / "sugar_scene" / "cameras.py"
        
        # 1. Patch the wrapper to accept new arguments
        if wrapper_path.exists():
            with open(wrapper_path, 'r') as f:
                content = f.read()
            
            if '--n_samples' not in content:
                # Add arguments to parser
                import re
                gpu_arg = "parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')"
                new_args = gpu_arg + "\n    parser.add_argument('--n_samples', type=int, default=1000000, help='Number of samples for SDF regularization.')" \
                                     "\n    parser.add_argument('--low_res', type=int, default=1, help='Downscale resolution factor.')"
                content = content.replace(gpu_arg, new_args)
                
                with open(wrapper_path, 'w') as f:
                    f.write(content)
                print("Patched train_coarse_density.py")

        # 2. Patch the trainer to use these arguments
        if trainer_path.exists():
            with open(trainer_path, 'r') as f:
                content = f.read()
            
            # Replace n_samples_for_sdf_regularization
            content = re.sub(
                r"n_samples_for_sdf_regularization\s*=\s*1_000_000.*",
                "n_samples_for_sdf_regularization = getattr(args, 'n_samples', 1_000_000)",
                content
            )
            # Replace downscale_resolution_factor
            content = re.sub(
                r"downscale_resolution_factor\s*=\s*1",
                "downscale_resolution_factor = getattr(args, 'low_res', 1)",
                content
            )
            
            with open(trainer_path, 'w') as f:
                f.write(content)
            print("Patched sugar_trainers/coarse_density.py")

        # 3. Patch CamerasWrapper to add missing rescale_output_resolution
        if cameras_path.exists():
            with open(cameras_path, 'r') as f:
                content = f.read()
            
            if 'def rescale_output_resolution(self, rescale_factor):' not in content:
                # Import fov2focal and focal2fov if possible, but they should be in scope
                # We'll add the method to CamerasWrapper
                old_method = "def get_spatial_extent(self):"
                new_method = """def rescale_output_resolution(self, rescale_factor):
        from sugar_utils.graphics_utils import fov2focal
        for gs_camera in self.gs_cameras:
            gs_camera.image_width = round(gs_camera.image_width * rescale_factor)
            gs_camera.image_height = round(gs_camera.image_height * rescale_factor)
        self.width = torch.tensor(np.array([gs_camera.image_width for gs_camera in self.gs_cameras]), dtype=torch.int).to(self.device)
        self.height = torch.tensor(np.array([gs_camera.image_height for gs_camera in self.gs_cameras]), dtype=torch.int).to(self.device)
        self.fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in self.gs_cameras])).to(self.device)
        self.fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in self.gs_cameras])).to(self.device)
        self.cx = self.width / 2.
        self.cy = self.height / 2.

    def get_spatial_extent(self):"""
                content = content.replace(old_method, new_method)
                
                with open(cameras_path, 'w') as f:
                    f.write(content)
                print("Patched sugar_scene/cameras.py")

    def train_sugar(self, colmap_dir: Path):
        """Train SuGaR model with surface alignment"""
        print("Training SuGaR (Surface-Aligned Gaussian Reconstruction)...")
        
        sugar_config = {
            'source_path': str(colmap_dir),
            'model_path': str(self.output_dir / 'sugar_output'),
            'iterations': self.config.get('iterations', 7000),
            'sh_degree': 3,
        }
        
        torch.cuda.empty_cache()
        
        # Apply patches
        self.patch_sugar()
        
        # Stage 1: Vanilla 3DGS training
        print("Stage 1: Vanilla 3DGS training...")
        vanilla_dir = self.output_dir / '3dgs_vanilla'
        cmd_vanilla = [
            'python', '/workspace/SuGaR/gaussian_splatting/train.py',
            '-s', str(colmap_dir),
            '-m', str(vanilla_dir),
            '--iterations', str(sugar_config['iterations']),
            '--quiet'
        ]
        subprocess.run(cmd_vanilla, check=True)

        torch.cuda.empty_cache()

        # Stage 2: SuGaR coarse refinement
        print("Stage 2: SuGaR surface alignment...")
        sugar_dir = self.output_dir / 'sugar_output'
        
        # Extract memory settings from config
        sugar_opts = self.config.get('sugar_settings', {})
        n_samples = sugar_opts.get('n_samples_for_sdf_regularization', 1000000)
        low_res = sugar_opts.get('downscale_resolution_factor', 1)
        
        cmd_sugar = [
            'python', '/workspace/SuGaR/train_coarse_density.py',
            '-s', str(colmap_dir),
            '-c', str(vanilla_dir) + '/',
            '-o', str(sugar_dir),
            '-i', str(sugar_config['iterations']),
            '--n_samples', str(n_samples),
            '--low_res', str(low_res)
        ]
        subprocess.run(cmd_sugar, check=True)

        print(f"SuGaR model trained and saved to {self.output_dir}")
        
        # Locate best checkpoint
        pt_files = sorted(list(sugar_dir.glob('**/*.pt')))
        if not pt_files:
            raise FileNotFoundError("SuGaR failed to produce .pt checkpoints.")
        
        best_pt = pt_files[-1]
        ply_path = sugar_dir / 'sugar_refined.ply'
        self.export_pt_to_ply(best_pt, ply_path)
        return sugar_dir

    def export_pt_to_ply(self, pt_path: Path, ply_path: Path):
        """Export SuGaR .pt checkpoint to a high-fidelity .ply file for Segmentation/Physics"""
        print(f"Exporting {pt_path.name} to {ply_path.name}...")
        
        checkpoint = torch.load(pt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Extract parameters
        xyz = state_dict.get('_points', None)
        if xyz is None:
            xyz = next((v for k, v in state_dict.items() if k.endswith('_points')), None)
        
        if xyz is None:
            raise ValueError(f"Could not find points in {pt_path}")

        n_points = xyz.shape[0]
        
        # 1. Colors from SH
        sh = state_dict.get('_sh_coordinates_dc', None)
        if sh is not None:
            if sh.ndim == 3:
                sh = sh[:, 0, :]
            C0 = 1 / (2 * np.sqrt(np.pi))
            colors = (sh * C0 + 0.5).clamp(0, 1) * 255
        else:
            colors = torch.ones((n_points, 3)) * 200

        # 2. Opacity
        opacity_raw = state_dict.get('all_densities', None)
        opacity = torch.sigmoid(opacity_raw) if opacity_raw is not None else torch.ones((n_points, 1)) * 0.9

        # 3. Scales
        scales = state_dict.get('_scales', None)
        if scales is None:
            scales = torch.ones((n_points, 3)) * -2.0
        
        # 4. Rotations
        quats = state_dict.get('_quaternions', None)
        if quats is None:
            quats = torch.zeros((n_points, 4))
            quats[:, 0] = 1.0

        xyz_np = xyz.detach().cpu().numpy()
        colors_np = colors.detach().cpu().numpy().astype(np.uint8)
        opacity_np = opacity.detach().cpu().numpy()
        scales_np = scales.detach().cpu().numpy()
        quats_np = quats.detach().cpu().numpy()

        vertex_data = [
            (xyz_np[i, 0], xyz_np[i, 1], xyz_np[i, 2],
                opacity_np[i, 0],
                colors_np[i, 0], colors_np[i, 1], colors_np[i, 2],
                scales_np[i, 0], scales_np[i, 1], scales_np[i, 2],
                quats_np[i, 0], quats_np[i, 1], quats_np[i, 2], quats_np[i, 3])
            for i in range(n_points)
        ]

        vertex_format = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('opacity', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ]
        
        el = PlyElement.describe(np.array(vertex_data, dtype=vertex_format), 'vertex')
        PlyData([el]).write(str(ply_path))
        print(f"High-fidelity export saved to {ply_path}")

    def validate_output(self, model_path: Path):
        """Validate the 3DGS output quality"""
        print("Validating output...")
        ply_files = list(model_path.glob('*.ply'))
        if not ply_files:
            raise RuntimeError("No .ply file found in output!")
        
        ply_path = ply_files[0]
        plydata = PlyData.read(str(ply_path))
        n_gaussians = len(plydata['vertex'])
        
        print(f"Found {ply_path.name} with {n_gaussians:,} gaussians")
        
        # Save metadata
        metadata = {
            'n_gaussians': n_gaussians,
            'ply_path': str(ply_path),
            'config': self.config
        }
        with open(self.output_dir / 'reconstruction_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return ply_path
    
    def run(self):
        """Execute the full Reconstruction pipeline"""
        print("=" * 60)
        print("RECONSTRUCTION: Neural Reconstruction & Surface Alignment")
        print("=" * 60)
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available!")
        
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.2f} GB)")
        
        frames_dir = self.extract_frames()
        colmap_dir = self.run_colmap(frames_dir)
        model_path = self.train_sugar(colmap_dir)
        ply_path = self.validate_output(model_path)
        
        print("\n" + "=" * 60)
        print("RECONSTRUCTION COMPLETE!")
        print("=" * 60)
        print(f"Metadata: {self.output_dir / 'reconstruction_metadata.json'}")
        print(f"Output: {ply_path}")
        print("\nNext: Run segmentation for object isolation")


def main():
    parser = argparse.ArgumentParser(description='Reconstruction: Neural Reconstruction')
    parser.add_argument('--config', type=str, default='/workspace/configs/pipeline_config.json',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    pipeline = ReconstructionPipeline(args.config)
    pipeline.run()


if __name__ == '__main__':
    main()
