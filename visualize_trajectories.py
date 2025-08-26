#!/usr/bin/env python3
"""
Trajectory Visualization Tool for DINO World Model Planning

This script provides an interactive way to visualize planning trajectories 
from the plan_outputs directory.

Usage:
    python visualize_trajectories.py [--run_dir <path>] [--list] [--web]
    
Examples:
    # List all available runs
    python visualize_trajectories.py --list
    
    # View a specific run directory
    python visualize_trajectories.py --run_dir plan_outputs/20250826173831_point_maze_gH5
    
    # Start web interface for interactive browsing
    python visualize_trajectories.py --web
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from IPython.display import Video, Image, display
    HAS_DISPLAY = True
except ImportError:
    HAS_DISPLAY = False
    print("Warning: matplotlib/IPython not available. Limited visualization capabilities.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: opencv-python not available. Video playback may be limited.")


class TrajectoryVisualizer:
    """Class to visualize planning trajectories from plan_outputs"""
    
    def __init__(self, plan_outputs_dir: str = "plan_outputs"):
        self.plan_outputs_dir = Path(plan_outputs_dir)
        if not self.plan_outputs_dir.exists():
            raise FileNotFoundError(f"Plan outputs directory not found: {plan_outputs_dir}")
    
    def list_runs(self) -> List[Tuple[str, Dict]]:
        """List all available planning runs with their metadata"""
        runs = []
        
        for run_dir in self.plan_outputs_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            # Parse run directory name
            run_name = run_dir.name
            run_info = self._parse_run_name(run_name)
            
            # Load logs if available
            logs_path = run_dir / "logs.json"
            if logs_path.exists():
                try:
                    with open(logs_path, 'r') as f:
                        lines = f.readlines()
                        logs = []
                        for line in lines:
                            if line.strip():
                                logs.append(json.loads(line.strip()))
                        run_info['logs'] = logs
                        if logs:
                            # Get final evaluation metrics
                            final_log = logs[-1] if logs else {}
                            run_info['success_rate'] = final_log.get('final_eval/success_rate', 'N/A')
                            run_info['mean_state_dist'] = final_log.get('final_eval/mean_state_dist', 'N/A')
                except Exception as e:
                    run_info['logs'] = None
                    run_info['error'] = str(e)
            
            # Check for available files
            run_info['files'] = {
                'videos': list(run_dir.glob("*.mp4")),
                'images': list(run_dir.glob("*.png")),
                'pkl_files': list(run_dir.glob("*.pkl")),
                'logs': logs_path if logs_path.exists() else None
            }
            
            runs.append((run_name, run_info))
        
        return sorted(runs, key=lambda x: x[0], reverse=True)  # Sort by timestamp, newest first
    
    def _parse_run_name(self, run_name: str) -> Dict:
        """Parse information from run directory name"""
        parts = run_name.split('_')
        info = {'run_name': run_name}
        
        if len(parts) >= 3:
            info['timestamp'] = parts[0]
            info['env'] = parts[1]
            info['model'] = parts[2]
            if len(parts) > 3:
                # Extract goal horizon if present
                for part in parts[3:]:
                    if part.startswith('gH'):
                        try:
                            info['goal_horizon'] = int(part[2:])
                        except ValueError:
                            pass
        
        return info
    
    def display_run_summary(self, run_name: str, run_info: Dict):
        """Display a summary of a planning run"""
        print(f"\n{'='*60}")
        print(f"Planning Run: {run_name}")
        print(f"{'='*60}")
        
        # Basic info
        print(f"Timestamp: {run_info.get('timestamp', 'N/A')}")
        print(f"Environment: {run_info.get('env', 'N/A')}")
        print(f"Model: {run_info.get('model', 'N/A')}")
        print(f"Goal Horizon: {run_info.get('goal_horizon', 'N/A')}")
        
        # Performance metrics
        if 'success_rate' in run_info:
            print(f"\nPerformance Metrics:")
            print(f"  Success Rate: {run_info['success_rate']:.3f}" if isinstance(run_info['success_rate'], float) else f"  Success Rate: {run_info['success_rate']}")
            print(f"  Mean State Distance: {run_info['mean_state_dist']:.3f}" if isinstance(run_info['mean_state_dist'], float) else f"  Mean State Distance: {run_info['mean_state_dist']}")
        
        # Available files
        files = run_info.get('files', {})
        print(f"\nAvailable Files:")
        print(f"  Videos: {len(files.get('videos', []))}")
        print(f"  Images: {len(files.get('images', []))}")
        print(f"  Pickle files: {len(files.get('pkl_files', []))}")
        print(f"  Logs: {'Yes' if files.get('logs') else 'No'}")
        
        return files
    
    def show_videos(self, run_dir: Path, max_videos: int = 3):
        """Display videos from a run"""
        video_files = list(run_dir.glob("*.mp4"))
        
        if not video_files:
            print("No videos found in this run.")
            return
        
        print(f"\nFound {len(video_files)} videos:")
        for i, video_path in enumerate(video_files[:max_videos]):
            print(f"  {i+1}. {video_path.name}")
            
            if HAS_DISPLAY:
                try:
                    # Try to display video if in Jupyter
                    display(Video(str(video_path), width=400))
                except:
                    print(f"    Video path: {video_path}")
            else:
                print(f"    Video path: {video_path}")
        
        if len(video_files) > max_videos:
            print(f"  ... and {len(video_files) - max_videos} more videos")
    
    def show_images(self, run_dir: Path):
        """Display images from a run"""
        image_files = list(run_dir.glob("*.png"))
        
        if not image_files:
            print("No images found in this run.")
            return
        
        print(f"\nFound {len(image_files)} images:")
        
        if HAS_DISPLAY:
            fig, axes = plt.subplots(1, min(len(image_files), 3), figsize=(15, 5))
            if len(image_files) == 1:
                axes = [axes]
            
            for i, img_path in enumerate(image_files[:3]):
                try:
                    img = mpimg.imread(str(img_path))
                    if i < len(axes):
                        axes[i].imshow(img)
                        axes[i].set_title(img_path.name, fontsize=10)
                        axes[i].axis('off')
                except Exception as e:
                    print(f"Error loading {img_path.name}: {e}")
            
            plt.tight_layout()
            plt.show()
        else:
            for img_path in image_files:
                print(f"  {img_path.name}: {img_path}")
    
    def show_logs(self, run_dir: Path):
        """Display detailed logs from a run"""
        logs_path = run_dir / "logs.json"
        
        if not logs_path.exists():
            print("No logs found for this run.")
            return
        
        try:
            with open(logs_path, 'r') as f:
                lines = f.readlines()
                
            print(f"\nDetailed Logs:")
            print("-" * 40)
            
            for i, line in enumerate(lines):
                if line.strip():
                    log_entry = json.loads(line.strip())
                    print(f"Step {i+1}:")
                    for key, value in log_entry.items():
                        if isinstance(value, float):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
                    print()
                    
        except Exception as e:
            print(f"Error reading logs: {e}")
    
    def show_targets(self, run_dir: Path):
        """Display planning targets information"""
        pkl_path = run_dir / "plan_targets.pkl"
        
        if not pkl_path.exists():
            print("No planning targets file found.")
            return
        
        try:
            with open(pkl_path, 'rb') as f:
                targets = pickle.load(f)
            
            print(f"\nPlanning Targets:")
            print("-" * 40)
            print(f"Goal Horizon: {targets.get('goal_H', 'N/A')}")
            
            if 'obs_0' in targets:
                obs_0 = targets['obs_0']
                print(f"Initial Observations:")
                for key, value in obs_0.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                    else:
                        print(f"  {key}: {type(value)}")
            
            if 'obs_g' in targets:
                obs_g = targets['obs_g']
                print(f"Goal Observations:")
                for key, value in obs_g.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                    else:
                        print(f"  {key}: {type(value)}")
            
            if 'state_0' in targets and hasattr(targets['state_0'], 'shape'):
                print(f"Initial States: shape {targets['state_0'].shape}")
                
            if 'state_g' in targets and hasattr(targets['state_g'], 'shape'):
                print(f"Goal States: shape {targets['state_g'].shape}")
                
        except Exception as e:
            print(f"Error reading planning targets: {e}")
    
    def visualize_run(self, run_dir: str, show_videos: bool = True, show_images: bool = True, 
                     show_logs: bool = True, show_targets: bool = True):
        """Visualize a complete planning run"""
        run_path = Path(run_dir)
        if not run_path.exists():
            # Try relative to plan_outputs_dir
            run_path = self.plan_outputs_dir / run_dir
            if not run_path.exists():
                raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        # Display summary
        run_name = run_path.name
        run_info = self._parse_run_name(run_name)
        
        # Load logs for metrics
        logs_path = run_path / "logs.json"
        if logs_path.exists():
            try:
                with open(logs_path, 'r') as f:
                    lines = f.readlines()
                    logs = []
                    for line in lines:
                        if line.strip():
                            logs.append(json.loads(line.strip()))
                    if logs:
                        final_log = logs[-1]
                        run_info['success_rate'] = final_log.get('final_eval/success_rate', 'N/A')
                        run_info['mean_state_dist'] = final_log.get('final_eval/mean_state_dist', 'N/A')
            except:
                pass
        
        files = self.display_run_summary(run_name, run_info)
        
        # Show different components based on flags
        if show_videos:
            self.show_videos(run_path)
        
        if show_images:
            self.show_images(run_path)
        
        if show_logs:
            self.show_logs(run_path)
        
        if show_targets:
            self.show_targets(run_path)


def start_web_interface(visualizer: TrajectoryVisualizer):
    """Start a simple web interface for browsing runs"""
    try:
        import http.server
        import socketserver
        import webbrowser
        import json
        from urllib.parse import parse_qs, urlparse
    except ImportError:
        print("Web interface requires standard library modules that are not available.")
        return
    
    print("Web interface not implemented yet. Use command line interface instead.")
    print("Available commands:")
    print("  --list: List all runs")
    print("  --run_dir <path>: Visualize specific run")


def main():
    parser = argparse.ArgumentParser(description='Visualize planning trajectories')
    parser.add_argument('--plan_outputs_dir', default='plan_outputs',
                       help='Directory containing planning outputs (default: plan_outputs)')
    parser.add_argument('--list', action='store_true',
                       help='List all available planning runs')
    parser.add_argument('--run_dir', type=str,
                       help='Specific run directory to visualize')
    parser.add_argument('--web', action='store_true',
                       help='Start web interface (not implemented yet)')
    parser.add_argument('--no-videos', action='store_true',
                       help='Skip video display')
    parser.add_argument('--no-images', action='store_true',
                       help='Skip image display')
    parser.add_argument('--no-logs', action='store_true',
                       help='Skip detailed logs')
    parser.add_argument('--no-targets', action='store_true',
                       help='Skip planning targets info')
    
    args = parser.parse_args()
    
    try:
        visualizer = TrajectoryVisualizer(args.plan_outputs_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    if args.web:
        start_web_interface(visualizer)
        return 0
    
    if args.list:
        print("Available Planning Runs:")
        print("=" * 60)
        
        runs = visualizer.list_runs()
        if not runs:
            print("No planning runs found.")
            return 0
        
        for i, (run_name, run_info) in enumerate(runs, 1):
            print(f"{i:2d}. {run_name}")
            if 'success_rate' in run_info:
                sr = run_info['success_rate']
                sr_str = f"{sr:.3f}" if isinstance(sr, float) else str(sr)
                print(f"     Success Rate: {sr_str}")
            files = run_info.get('files', {})
            video_count = len(files.get('videos', []))
            image_count = len(files.get('images', []))
            print(f"     Files: {video_count} videos, {image_count} images")
            print()
        
        print(f"\nTo visualize a specific run, use:")
        print(f"python {sys.argv[0]} --run_dir <run_name>")
        return 0
    
    if args.run_dir:
        try:
            visualizer.visualize_run(
                args.run_dir,
                show_videos=not args.no_videos,
                show_images=not args.no_images,
                show_logs=not args.no_logs,
                show_targets=not args.no_targets
            )
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
        return 0
    
    # Interactive mode - show list and prompt for selection
    print("DINO World Model - Trajectory Visualizer")
    print("=" * 50)
    
    runs = visualizer.list_runs()
    if not runs:
        print("No planning runs found.")
        return 0
    
    print("\nAvailable runs (most recent first):")
    for i, (run_name, run_info) in enumerate(runs[:10], 1):  # Show top 10
        sr = run_info.get('success_rate', 'N/A')
        sr_str = f"{sr:.3f}" if isinstance(sr, float) else str(sr)
        print(f"{i:2d}. {run_name} (Success: {sr_str})")
    
    if len(runs) > 10:
        print(f"    ... and {len(runs) - 10} more runs")
    
    print(f"\nUsage examples:")
    print(f"  python {sys.argv[0]} --list")
    print(f"  python {sys.argv[0]} --run_dir {runs[0][0]}")
    
    return 0


if __name__ == "__main__":
    exit(main())