#!/usr/bin/env python3
"""
DINO World Model Planning Visualizer - Actually Understand WTF is Going On

This script helps you understand what your planning agent is actually doing:
- Shows the planning problem (start → goal)
- Compares what the agent THINKS will happen vs what ACTUALLY happens
- Highlights success/failure cases
- Shows planning metrics and explains what they mean
- Creates side-by-side videos and analysis

Usage:
    python visualize_planning.py [run_name]
    python visualize_planning.py --latest
    python visualize_planning.py --compare
    python visualize_planning.py --analyze-failures
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import shutil

class PlanningAnalyzer:
    """Analyze and visualize planning runs to understand agent behavior"""
    
    def __init__(self, plan_outputs_dir: str = "plan_outputs"):
        self.plan_outputs_dir = Path(plan_outputs_dir)
        if not self.plan_outputs_dir.exists():
            print(f"❌ Plan outputs directory not found: {plan_outputs_dir}")
            print(f"   Make sure you've run some planning experiments first!")
            sys.exit(1)
    
    def get_runs(self, sort_by_time: bool = True) -> List[Tuple[str, Path, Dict]]:
        """Get all planning runs with their metadata"""
        runs = []
        
        for run_dir in self.plan_outputs_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith('.'):
                continue
            
            metadata = self._load_run_metadata(run_dir)
            runs.append((run_dir.name, run_dir, metadata))
        
        if sort_by_time:
            runs.sort(key=lambda x: x[0], reverse=True)  # Newest first
        
        return runs
    
    def _load_run_metadata(self, run_dir: Path) -> Dict:
        """Load all metadata for a run"""
        metadata = {
            'timestamp': None,
            'env': 'unknown',
            'goal_horizon': 'unknown',
            'success_rate': None,
            'files': {},
            'metrics': {},
            'targets': None,
            'config': None
        }
        
        # Parse run name
        parts = run_dir.name.split('_')
        if len(parts) >= 3:
            metadata['timestamp'] = parts[0]
            metadata['env'] = parts[1]
            for part in parts:
                if part.startswith('gH'):
                    try:
                        metadata['goal_horizon'] = int(part[2:])
                    except:
                        pass
        
        # Find files
        metadata['files'] = {
            'videos': sorted(run_dir.glob("*.mp4")),
            'images': sorted(run_dir.glob("*.png")),
            'logs': run_dir / "logs.json" if (run_dir / "logs.json").exists() else None,
            'targets': run_dir / "plan_targets.pkl" if (run_dir / "plan_targets.pkl").exists() else None,
            'config': next(run_dir.glob("*.yaml"), None)
        }
        
        # Load performance metrics
        if metadata['files']['logs']:
            try:
                with open(metadata['files']['logs'], 'r') as f:
                    logs = []
                    for line in f:
                        if line.strip():
                            logs.append(json.loads(line.strip()))
                    
                    if logs:
                        final_metrics = logs[-1]
                        metadata['success_rate'] = final_metrics.get('final_eval/success_rate', None)
                        metadata['metrics'] = final_metrics
            except Exception as e:
                print(f"⚠️  Error loading logs for {run_dir.name}: {e}")
        
        # Load planning targets
        if metadata['files']['targets']:
            try:
                with open(metadata['files']['targets'], 'rb') as f:
                    metadata['targets'] = pickle.load(f)
            except Exception as e:
                print(f"⚠️  Error loading targets for {run_dir.name}: {e}")
        
        return metadata
    
    def show_run_overview(self):
        """Show overview of all planning runs"""
        runs = self.get_runs()
        
        if not runs:
            print("🤷 No planning runs found!")
            print("   Run some planning experiments first with: python plan.py")
            return
        
        print("🎯 PLANNING RUNS OVERVIEW")
        print("=" * 60)
        print(f"Found {len(runs)} planning runs\n")
        
        success_runs = []
        failure_runs = []
        
        for i, (name, path, meta) in enumerate(runs[:10]):  # Show latest 10
            sr = meta.get('success_rate')
            sr_str = f"{sr:.1%}" if sr is not None else "???"
            
            status = "✅" if sr == 1.0 else "❌" if sr == 0.0 else "🔶" if sr is not None else "❓"
            
            print(f"{i+1:2d}. {status} {name}")
            print(f"     Environment: {meta['env']} | Goal Horizon: {meta['goal_horizon']} | Success: {sr_str}")
            
            videos = len(meta['files']['videos'])
            images = len(meta['files']['images'])
            print(f"     Files: {videos} videos, {images} images")
            
            if sr is not None:
                if sr >= 0.8:
                    success_runs.append((name, sr))
                elif sr <= 0.2:
                    failure_runs.append((name, sr))
            print()
        
        if len(runs) > 10:
            print(f"... and {len(runs) - 10} more runs")
        
        print("\n📊 QUICK ANALYSIS:")
        if success_runs:
            print(f"✅ High success runs ({len(success_runs)}): {', '.join([f'{n} ({sr:.1%})' for n, sr in success_runs[:3]])}")
        if failure_runs:
            print(f"❌ Low success runs ({len(failure_runs)}): {', '.join([f'{n} ({sr:.1%})' for n, sr in failure_runs[:3]])}")
        
        print(f"\n🔍 TO DIVE DEEPER:")
        if runs:
            latest = runs[0][0]
            print(f"   python visualize_planning.py {latest}")
            print(f"   python visualize_planning.py --compare")
            print(f"   python visualize_planning.py --analyze-failures")
    
    def analyze_run(self, run_name: str):
        """Deep dive analysis of a specific run"""
        runs = {name: (path, meta) for name, path, meta in self.get_runs()}
        
        if run_name not in runs:
            print(f"❌ Run '{run_name}' not found!")
            print(f"Available runs: {list(runs.keys())[:5]}...")
            return
        
        run_path, metadata = runs[run_name]
        
        print(f"🔬 DEEP DIVE ANALYSIS: {run_name}")
        print("=" * 60)
        
        # Basic info
        self._print_run_basics(metadata)
        
        # Planning problem setup
        self._analyze_planning_problem(metadata)
        
        # Performance analysis  
        self._analyze_performance(metadata)
        
        # File analysis
        self._analyze_files(run_path, metadata)
        
        # What to do next
        self._suggest_next_steps(run_path, metadata)
    
    def _print_run_basics(self, metadata: Dict):
        """Print basic run information"""
        print(f"📋 RUN DETAILS:")
        print(f"   Environment: {metadata['env']}")
        print(f"   Goal Horizon: {metadata['goal_horizon']} steps")
        print(f"   Timestamp: {metadata['timestamp']}")
        
        sr = metadata.get('success_rate')
        if sr is not None:
            status = "SUCCESS! 🎉" if sr >= 0.8 else "FAILURE 💥" if sr <= 0.2 else "MIXED RESULTS 🤔"
            print(f"   Success Rate: {sr:.1%} - {status}")
        else:
            print(f"   Success Rate: Unknown ❓")
        print()
    
    def _analyze_planning_problem(self, metadata: Dict):
        """Analyze the planning problem setup"""
        print(f"🎯 PLANNING PROBLEM:")
        
        targets = metadata.get('targets')
        if targets is None:
            print("   ❓ No planning targets found - can't analyze problem setup")
            return
        
        # Analyze initial and goal conditions
        if 'obs_0' in targets and 'obs_g' in targets:
            obs_0 = targets['obs_0']
            obs_g = targets['obs_g']
            
            print(f"   📍 Start → Goal planning problem")
            print(f"   🔢 Planning for {len(obs_0.get('visual', []))} parallel episodes")
            
            if 'visual' in obs_0:
                visual_shape = obs_0['visual'].shape
                print(f"   🖼️  Visual observations: {visual_shape}")
            
            if 'proprio' in obs_0:
                proprio_shape = obs_0['proprio'].shape  
                print(f"   🤖 Proprioceptive observations: {proprio_shape}")
            
            goal_h = targets.get('goal_H', 'unknown')
            print(f"   ⏱️  Planning horizon: {goal_h} steps")
            
        else:
            print("   ❓ Incomplete target information")
        print()
    
    def _analyze_performance(self, metadata: Dict):
        """Analyze performance metrics"""
        print(f"📈 PERFORMANCE BREAKDOWN:")
        
        metrics = metadata.get('metrics', {})
        if not metrics:
            print("   ❓ No performance metrics found")
            return
        
        # Success rate analysis
        sr = metrics.get('final_eval/success_rate', metrics.get('mpc/success_rate'))
        if sr is not None:
            if sr >= 0.8:
                print(f"   ✅ Success Rate: {sr:.1%} - Agent is working well!")
            elif sr >= 0.5:
                print(f"   🔶 Success Rate: {sr:.1%} - Agent partially working")
            else:
                print(f"   ❌ Success Rate: {sr:.1%} - Agent struggling")
        
        # Distance analysis
        state_dist = metrics.get('final_eval/mean_state_dist', metrics.get('mpc/mean_state_dist'))
        if state_dist is not None:
            print(f"   📏 State Distance: {state_dist:.3f}")
            if state_dist < 0.1:
                print(f"      → Very close to goal! 🎯")
            elif state_dist < 0.5:
                print(f"      → Reasonably close to goal")
            else:
                print(f"      → Far from goal - planning issues? 🤔")
        
        # World model accuracy
        visual_emb_div = metrics.get('final_eval/mean_div_visual_emb', metrics.get('mpc/mean_div_visual_emb'))
        if visual_emb_div is not None:
            print(f"   🧠 World Model Visual Error: {visual_emb_div:.2f}")
            if visual_emb_div > 50:
                print(f"      → High error - world model predictions don't match reality! ⚠️")
            elif visual_emb_div > 20:
                print(f"      → Moderate error - some mismatch between prediction and reality")
            else:
                print(f"      → Low error - world model predictions are good! ✅")
        
        print()
    
    def _analyze_files(self, run_path: Path, metadata: Dict):
        """Analyze available files and what they show"""
        print(f"🎬 VISUALIZATION FILES:")
        
        files = metadata['files']
        
        # Videos
        videos = files.get('videos', [])
        if videos:
            print(f"   📹 Found {len(videos)} videos:")
            success_videos = [v for v in videos if 'success' in v.name]
            failure_videos = [v for v in videos if 'failure' in v.name]
            
            if success_videos:
                print(f"      ✅ {len(success_videos)} success videos")
                print(f"         Example: {success_videos[0].name}")
            if failure_videos:
                print(f"      ❌ {len(failure_videos)} failure videos")
                print(f"         Example: {failure_videos[0].name}")
            
            print(f"   💡 Open these videos to see what the agent actually did!")
            for video in videos[:2]:  # Show first 2
                print(f"      vlc '{video}' &")
        
        # Images
        images = files.get('images', [])
        if images:
            print(f"   🖼️  Found {len(images)} comparison images:")
            for img in images[:2]:
                print(f"      📸 {img.name}")
                print(f"         eog '{img}' &")
        
        if not videos and not images:
            print(f"   ❌ No visualization files found!")
            print(f"      This might be a failed run or missing decoder")
        
        print()
    
    def _suggest_next_steps(self, run_path: Path, metadata: Dict):
        """Suggest what to do next"""
        print(f"🚀 WHAT TO DO NEXT:")
        
        sr = metadata.get('success_rate')
        files = metadata['files']
        
        if sr is None:
            print(f"   1. Check if the run completed successfully")
            print(f"   2. Look at log files for errors")
        elif sr >= 0.8:
            print(f"   1. Great job! Agent is working well 🎉")
            print(f"   2. Try harder environments or longer horizons")
            print(f"   3. Compare with other successful runs")
        elif sr <= 0.2:
            print(f"   1. Debug why agent is failing:")
            print(f"      - Check world model predictions vs reality")
            print(f"      - Verify planning algorithm settings")
            print(f"      - Look at failure videos for patterns")
            print(f"   2. Try simpler problems first")
        else:
            print(f"   1. Mixed results - investigate both successes and failures")
            print(f"   2. Look for patterns in what works vs what doesn't")
        
        # Specific file recommendations
        if files.get('videos'):
            print(f"   📹 Watch videos to see agent behavior:")
            for video in files['videos'][:2]:
                print(f"      vlc '{video}' &")
        
        if files.get('images'):
            print(f"   🖼️  View comparison images:")
            for img in files['images'][:1]:
                print(f"      eog '{img}' &")
        
        print()
    
    def compare_runs(self):
        """Compare multiple runs to understand patterns"""
        runs = self.get_runs()
        
        if len(runs) < 2:
            print("🤷 Need at least 2 runs to compare")
            return
        
        print(f"🔄 COMPARING PLANNING RUNS")
        print("=" * 60)
        
        # Group by success rate
        successful = []
        failed = []
        mixed = []
        
        for name, path, meta in runs:
            sr = meta.get('success_rate')
            if sr is None:
                continue
            elif sr >= 0.8:
                successful.append((name, sr, meta))
            elif sr <= 0.2:
                failed.append((name, sr, meta))
            else:
                mixed.append((name, sr, meta))
        
        print(f"📊 SUCCESS PATTERNS:")
        if successful:
            print(f"   ✅ {len(successful)} successful runs:")
            for name, sr, meta in successful[:3]:
                env = meta.get('env', 'unknown')
                horizon = meta.get('goal_horizon', '?')
                print(f"      {name}: {sr:.1%} (env={env}, horizon={horizon})")
        else:
            print(f"   😰 No fully successful runs found!")
        
        print(f"\n💥 FAILURE PATTERNS:")
        if failed:
            print(f"   ❌ {len(failed)} failed runs:")
            for name, sr, meta in failed[:3]:
                env = meta.get('env', 'unknown')  
                horizon = meta.get('goal_horizon', '?')
                print(f"      {name}: {sr:.1%} (env={env}, horizon={horizon})")
        else:
            print(f"   🎉 No complete failures!")
        
        if mixed:
            print(f"\n🔶 MIXED RESULTS:")
            print(f"   📊 {len(mixed)} partially successful runs")
        
        # Environment analysis
        env_performance = {}
        horizon_performance = {}
        
        for name, path, meta in runs:
            sr = meta.get('success_rate')
            if sr is None:
                continue
            
            env = meta.get('env', 'unknown')
            horizon = meta.get('goal_horizon', 'unknown')
            
            if env not in env_performance:
                env_performance[env] = []
            env_performance[env].append(sr)
            
            if horizon not in horizon_performance:
                horizon_performance[horizon] = []
            horizon_performance[horizon].append(sr)
        
        print(f"\n🌍 ENVIRONMENT ANALYSIS:")
        for env, success_rates in env_performance.items():
            avg_sr = sum(success_rates) / len(success_rates)
            print(f"   {env}: {avg_sr:.1%} average success ({len(success_rates)} runs)")
        
        print(f"\n⏱️  HORIZON ANALYSIS:")
        for horizon, success_rates in horizon_performance.items():
            avg_sr = sum(success_rates) / len(success_rates)  
            print(f"   {horizon} steps: {avg_sr:.1%} average success ({len(success_rates)} runs)")
        
        print(f"\n🔍 INSIGHTS:")
        if env_performance:
            best_env = max(env_performance.items(), key=lambda x: sum(x[1])/len(x[1]))
            worst_env = min(env_performance.items(), key=lambda x: sum(x[1])/len(x[1]))
            print(f"   🏆 Best environment: {best_env[0]} ({sum(best_env[1])/len(best_env[1]):.1%})")
            print(f"   💥 Worst environment: {worst_env[0]} ({sum(worst_env[1])/len(worst_env[1]):.1%})")
        
        print()
    
    def analyze_failures(self):
        """Focus on understanding failure cases"""
        runs = self.get_runs()
        failed_runs = [(name, path, meta) for name, path, meta in runs 
                      if meta.get('success_rate') is not None and meta['success_rate'] <= 0.3]
        
        if not failed_runs:
            print("🎉 No major failures found! All runs are doing reasonably well.")
            return
        
        print(f"🚨 FAILURE ANALYSIS")
        print("=" * 60)
        print(f"Found {len(failed_runs)} runs with low success rates\n")
        
        for i, (name, path, meta) in enumerate(failed_runs[:5]):
            sr = meta['success_rate']
            print(f"{i+1}. ❌ {name} - {sr:.1%} success")
            
            # Analyze what went wrong
            metrics = meta.get('metrics', {})
            state_dist = metrics.get('final_eval/mean_state_dist')
            visual_emb_div = metrics.get('final_eval/mean_div_visual_emb')
            
            print(f"   📊 Diagnostics:")
            if state_dist is not None:
                if state_dist > 1.0:
                    print(f"      🎯 Very far from goal (dist={state_dist:.3f}) - planning not working")
                else:
                    print(f"      📏 State distance: {state_dist:.3f}")
            
            if visual_emb_div is not None:
                if visual_emb_div > 50:
                    print(f"      🧠 World model error: {visual_emb_div:.1f} - predictions way off!")
                else:
                    print(f"      🧠 World model error: {visual_emb_div:.1f}")
            
            # Check for failure videos
            failure_videos = [v for v in meta['files']['videos'] if 'failure' in v.name]
            if failure_videos:
                print(f"      🎬 Watch failure: vlc '{failure_videos[0]}' &")
            
            print()
        
        print(f"🔧 COMMON FAILURE PATTERNS TO CHECK:")
        print(f"   1. World model predictions don't match reality")
        print(f"   2. Planning algorithm not exploring well enough") 
        print(f"   3. Goal is too difficult for current horizon")
        print(f"   4. Environment dynamics too complex")
        print()


def main():
    parser = argparse.ArgumentParser(description='Understand what your planning agent is actually doing')
    parser.add_argument('run_name', nargs='?', help='Specific run to analyze')
    parser.add_argument('--latest', action='store_true', help='Analyze the most recent run')
    parser.add_argument('--compare', action='store_true', help='Compare multiple runs')
    parser.add_argument('--analyze-failures', action='store_true', help='Focus on failure analysis')
    parser.add_argument('--plan-outputs-dir', default='plan_outputs', help='Directory with planning outputs')
    
    args = parser.parse_args()
    
    analyzer = PlanningAnalyzer(args.plan_outputs_dir)
    
    if args.latest:
        runs = analyzer.get_runs()
        if runs:
            latest_run = runs[0][0]
            print(f"🎯 Analyzing latest run: {latest_run}\n")
            analyzer.analyze_run(latest_run)
        else:
            print("No runs found!")
    elif args.compare:
        analyzer.compare_runs()
    elif args.analyze_failures:
        analyzer.analyze_failures()
    elif args.run_name:
        analyzer.analyze_run(args.run_name)
    else:
        analyzer.show_run_overview()


if __name__ == "__main__":
    main()