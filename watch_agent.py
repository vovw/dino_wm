#!/usr/bin/env python3
"""
Watch Your Agent in Action - Open Videos and Images to See What's Happening

This script automatically opens the visualization files so you can watch
your planning agent in action and understand its behavior.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

def find_video_player():
    """Find available video player on the system"""
    players = ['vlc', 'mpv', 'mplayer', 'totem', 'xdg-open']
    for player in players:
        if subprocess.run(['which', player], capture_output=True).returncode == 0:
            return player
    return None

def find_image_viewer():
    """Find available image viewer on the system"""
    viewers = ['eog', 'feh', 'gpicview', 'xviewer', 'xdg-open']
    for viewer in viewers:
        if subprocess.run(['which', viewer], capture_output=True).returncode == 0:
            return viewer
    return None

def open_videos(video_files, max_videos=3):
    """Open video files with available player"""
    video_player = find_video_player()
    if not video_player:
        print("❌ No video player found! Install vlc, mpv, or another video player.")
        return False
    
    print(f"🎬 Opening videos with {video_player}...")
    
    # Separate success and failure videos
    success_videos = [v for v in video_files if 'success' in v.name]
    failure_videos = [v for v in video_files if 'failure' in v.name]
    
    videos_to_open = []
    
    if success_videos:
        print(f"   ✅ Found {len(success_videos)} success videos")
        videos_to_open.extend(success_videos[:max_videos//2 + 1])
    
    if failure_videos:
        print(f"   ❌ Found {len(failure_videos)} failure videos")
        videos_to_open.extend(failure_videos[:max_videos//2])
    
    # If no success/failure tagged videos, just take the first few
    if not videos_to_open:
        videos_to_open = video_files[:max_videos]
    
    for video in videos_to_open[:max_videos]:
        print(f"   📹 {video.name}")
        try:
            subprocess.Popen([video_player, str(video)], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            time.sleep(0.5)  # Small delay between opening videos
        except Exception as e:
            print(f"   ❌ Error opening {video.name}: {e}")
    
    return True

def open_images(image_files, max_images=3):
    """Open image files with available viewer"""
    image_viewer = find_image_viewer()
    if not image_viewer:
        print("❌ No image viewer found! Install eog, feh, or another image viewer.")
        return False
    
    print(f"🖼️  Opening images with {image_viewer}...")
    
    for image in image_files[:max_images]:
        print(f"   📸 {image.name}")
        try:
            subprocess.Popen([image_viewer, str(image)], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            time.sleep(0.3)
        except Exception as e:
            print(f"   ❌ Error opening {image.name}: {e}")
    
    return True

def watch_run(run_path: Path, videos_only=False, images_only=False):
    """Open all visualization files for a run"""
    print(f"👀 WATCHING: {run_path.name}")
    print("=" * 50)
    
    # Find video files
    videos = sorted(run_path.glob("*.mp4"))
    images = sorted(run_path.glob("*.png"))
    
    print(f"Found: {len(videos)} videos, {len(images)} images")
    
    if not videos and not images:
        print("❌ No visualization files found!")
        print("   This run might not have completed or lacks a decoder.")
        return False
    
    success = True
    
    if videos and not images_only:
        success &= open_videos(videos)
        if images and not videos_only:
            print()
            time.sleep(1)  # Brief pause between videos and images
    
    if images and not videos_only:
        success &= open_images(images)
    
    if success:
        print(f"\n✅ Opened visualization files!")
        print(f"💡 TIP: Look for:")
        print(f"   - Does the agent reach the goal?")
        print(f"   - Do the predicted visuals (right) match reality (left)?")
        print(f"   - Are there systematic failure patterns?")
    
    return success

def watch_latest():
    """Watch the most recent planning run"""
    plan_outputs = Path("plan_outputs")
    if not plan_outputs.exists():
        print("❌ No plan_outputs directory found!")
        return False
    
    runs = sorted([d for d in plan_outputs.iterdir() if d.is_dir()], 
                  key=lambda x: x.name, reverse=True)
    
    if not runs:
        print("❌ No planning runs found!")
        return False
    
    latest = runs[0]
    print(f"🎯 Watching latest run: {latest.name}\n")
    return watch_run(latest)

def watch_comparison():
    """Open videos from multiple runs for comparison"""
    plan_outputs = Path("plan_outputs")
    if not plan_outputs.exists():
        print("❌ No plan_outputs directory found!")
        return False
    
    runs = sorted([d for d in plan_outputs.iterdir() if d.is_dir()], 
                  key=lambda x: x.name, reverse=True)
    
    if len(runs) < 2:
        print("❌ Need at least 2 runs to compare!")
        return False
    
    print(f"🔄 COMPARISON MODE: Opening videos from multiple runs")
    print("=" * 60)
    
    # Take videos from top 3 most recent runs
    for i, run in enumerate(runs[:3]):
        print(f"\nRun {i+1}: {run.name}")
        videos = list(run.glob("*.mp4"))
        if videos:
            # Just open one video per run to avoid overwhelming
            success_videos = [v for v in videos if 'success' in v.name]
            video_to_open = success_videos[0] if success_videos else videos[0]
            
            video_player = find_video_player()
            if video_player:
                print(f"   📹 {video_to_open.name}")
                subprocess.Popen([video_player, str(video_to_open)], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
                time.sleep(1)
        else:
            print(f"   ❌ No videos found")
    
    print(f"\n💡 Compare the agent behaviors across different runs!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Watch your planning agent in action')
    parser.add_argument('run_name', nargs='?', help='Specific run to watch')
    parser.add_argument('--latest', action='store_true', help='Watch the most recent run')
    parser.add_argument('--compare', action='store_true', help='Compare multiple runs')
    parser.add_argument('--videos-only', action='store_true', help='Only open videos')
    parser.add_argument('--images-only', action='store_true', help='Only open images')
    parser.add_argument('--list', action='store_true', help='List available runs')
    
    args = parser.parse_args()
    
    plan_outputs = Path("plan_outputs")
    
    if args.list:
        if not plan_outputs.exists():
            print("❌ No plan_outputs directory found!")
            return 1
        
        runs = sorted([d for d in plan_outputs.iterdir() if d.is_dir()], 
                      key=lambda x: x.name, reverse=True)
        
        print(f"📁 Available runs ({len(runs)}):")
        for i, run in enumerate(runs[:10]):
            videos = len(list(run.glob("*.mp4")))
            images = len(list(run.glob("*.png")))
            print(f"   {i+1:2d}. {run.name} ({videos} videos, {images} images)")
        
        if len(runs) > 10:
            print(f"   ... and {len(runs) - 10} more")
        return 0
    
    if args.latest:
        return 0 if watch_latest() else 1
    
    if args.compare:
        return 0 if watch_comparison() else 1
    
    if args.run_name:
        run_path = plan_outputs / args.run_name
        if not run_path.exists():
            print(f"❌ Run '{args.run_name}' not found!")
            return 1
        
        return 0 if watch_run(run_path, args.videos_only, args.images_only) else 1
    
    # Default: show help and available runs
    print("👀 AGENT WATCHER - See What Your Planning Agent Actually Does")
    print("=" * 60)
    print("This tool opens videos and images so you can watch your agent in action!\n")
    
    if not plan_outputs.exists():
        print("❌ No plan_outputs directory found!")
        print("   Run some planning experiments first with: python plan.py")
        return 1
    
    runs = sorted([d for d in plan_outputs.iterdir() if d.is_dir()], 
                  key=lambda x: x.name, reverse=True)
    
    if not runs:
        print("❌ No planning runs found!")
        print("   Run some planning experiments first with: python plan.py")
        return 1
    
    print(f"Found {len(runs)} planning runs. Recent ones:")
    for i, run in enumerate(runs[:5]):
        videos = len(list(run.glob("*.mp4")))
        images = len(list(run.glob("*.png")))
        status = "✅" if videos > 0 else "❌"
        print(f"   {status} {run.name} ({videos} videos, {images} images)")
    
    print(f"\n🚀 QUICK START:")
    print(f"   python watch_agent.py --latest     # Watch newest run")
    print(f"   python watch_agent.py --compare    # Compare multiple runs")
    print(f"   python watch_agent.py {runs[0].name}  # Watch specific run")
    
    return 0

if __name__ == "__main__":
    exit(main())