"""
眠気検出システム - シンプル瞬きEAR可視化プログラム

JSONファイルから以下の2つのグラフを表示:
1. セッション全体のEAR連続時系列（いつ瞬きしているか分かる）
2. 各瞬きのEAR最小値の推移（瞬きの深さが分かる）

使い方:
    # 1つのJSONファイルを表示
    python simple_blink_visualizer.py data/sessions/20251112_143000_001_normal.json
    
    # 2つのJSONファイルを比較（正常 vs 眠気）
    python simple_blink_visualizer.py data/sessions/20251112_143000_001_normal.json data/sessions/20251112_150000_001_drowsy.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


class SimpleBlinkVisualizer:
    """Simple Blink EAR Visualizer Class"""
    
    # Blink region color
    BLINK_COLOR = 'lightcoral'
    EAR_LINE_COLOR = 'blue'
    MIN_MARKER_COLOR = 'red'
    THRESHOLD_COLOR = 'orange'
    
    def __init__(self, json_filepath):
        self.json_filepath = json_filepath
        self.session_data = None
        self.load_data()
    
    def load_data(self):
        """Load JSON file"""
        try:
            with open(self.json_filepath, 'r', encoding='utf-8') as f:
                self.session_data = json.load(f)
            
            session_id = self.session_data.get('session_id', 'N/A')
            user_id = self.session_data.get('user_id', 'N/A')
            label = self.session_data.get('label', 'N/A')
            label_str = 'Normal' if label == 0 else 'Drowsy'
            total_blinks = self.session_data.get('total_blinks', 0)
            
            print(f"✓ File loaded successfully")
            print(f"  Session ID: {session_id}")
            print(f"  User ID: {user_id}")
            print(f"  State: {label_str}")
            print(f"  Total blinks: {total_blinks}")
            
        except FileNotFoundError:
            print(f"❌ Error: File not found: {self.json_filepath}")
            raise
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON format: {self.json_filepath}")
            raise
    
    def plot_session_ear_timeline(self):
        """
        Display EAR time series for the entire session
        Shows when blinks occur at a glance
        """
        blinks = self.session_data.get('blinks', [])
        
        if not blinks:
            print("❌ Error: No blink data available")
            return
        
        # Concatenate EAR time series of all blinks
        all_times = []
        all_ears = []
        blink_regions = []  # Record blink intervals
        ear_min_points = []  # EAR minimum value points
        
        cumulative_time = 0.0
        
        for i, blink in enumerate(blinks):
            ear_timeseries = blink.get('ear_timeseries', [])
            if not ear_timeseries:
                continue
            
            stats = blink.get('statistics', {})
            blink_id = blink.get('blink_id', i + 1)
            
            # Set first timestamp to 0
            base_time = ear_timeseries[0]['timestamp']
            
            # Add EAR time series data
            blink_times = []
            blink_ears = []
            ear_min_value = stats.get('ear_min', 0)
            ear_min_time = None
            
            for point in ear_timeseries:
                t = cumulative_time + (point['timestamp'] - base_time)
                ear = point['ear']
                
                all_times.append(t)
                all_ears.append(ear)
                blink_times.append(t)
                blink_ears.append(ear)
                
                # Record the time of EAR minimum value
                if abs(ear - ear_min_value) < 0.001:
                    ear_min_time = t
            
            # Record blink interval
            if blink_times:
                blink_regions.append({
                    'start': blink_times[0],
                    'end': blink_times[-1],
                    'id': blink_id
                })
                
                # Record EAR minimum value point
                if ear_min_time is not None:
                    ear_min_points.append({
                        'time': ear_min_time,
                        'ear': ear_min_value,
                        'id': blink_id
                    })
            
            # Update cumulative time for next blink
            if ear_timeseries:
                cumulative_time += (ear_timeseries[-1]['timestamp'] - base_time)
                
                # Add interval between blinks (if there's a next blink)
                if i < len(blinks) - 1:
                    interval = stats.get('interval', 0.5)
                    cumulative_time += interval
        
        # Create graph
        fig, axes = plt.subplots(2, 1, figsize=(16, 11))
        
        session_id = self.session_data.get('session_id', 'N/A')
        user_id = self.session_data.get('user_id', 'N/A')
        label = self.session_data.get('label', 'N/A')
        label_str = 'Normal State' if label == 0 else 'Drowsy State'
        
        fig.suptitle(f'Blink EAR Visualization\nSession: {session_id} (User: {user_id}, {label_str})', 
                     fontsize=14, fontweight='bold')
        
        # Graph 1: EAR continuous time series for the entire session
        ax1 = axes[0]
        
        # Plot EAR time series
        ax1.plot(all_times, all_ears, color=self.EAR_LINE_COLOR, linewidth=1.5, 
                label='EAR', alpha=0.8, zorder=1)
        
        # Threshold line
        ax1.axhline(y=0.21, color=self.THRESHOLD_COLOR, linestyle='--', 
                   linewidth=2, label='Threshold (0.21)', alpha=0.7, zorder=2)
        
        # Color blink intervals
        for region in blink_regions:
            ax1.axvspan(region['start'], region['end'], 
                       color=self.BLINK_COLOR, alpha=0.3, zorder=0)
            
            # Display blink ID
            mid_time = (region['start'] + region['end']) / 2
            ax1.text(mid_time, 0.35, f"#{region['id']}", 
                    ha='center', va='bottom', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Mark EAR minimum value points
        for point in ear_min_points:
            ax1.plot(point['time'], point['ear'], 
                    marker='o', markersize=8, color=self.MIN_MARKER_COLOR, 
                    zorder=5)
            ax1.text(point['time'], point['ear'] - 0.02, 
                    f"{point['ear']:.3f}", 
                    ha='center', va='top', fontsize=8, color=self.MIN_MARKER_COLOR)
        
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('EAR Value', fontsize=12)
        ax1.set_title('EAR Time Series for Entire Session (When Blinks Occur)', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.05, 0.40)
        
        # Graph 2: EAR minimum value trend for each blink
        ax2 = axes[1]
        
        blink_ids = [p['id'] for p in ear_min_points]
        ear_mins = [p['ear'] for p in ear_min_points]
        
        # Line graph + markers
        ax2.plot(blink_ids, ear_mins, color='darkblue', linewidth=2, 
                marker='o', markersize=8, label='EAR Minimum', zorder=2)
        
        # Display values on each point
        for bid, ear_min in zip(blink_ids, ear_mins):
            ax2.text(bid, ear_min + 0.005, f"{ear_min:.3f}", 
                    ha='center', va='bottom', fontsize=9)
        
        # Threshold line
        ax2.axhline(y=0.21, color=self.THRESHOLD_COLOR, linestyle='--', 
                   linewidth=2, label='Threshold (0.21)', alpha=0.7)
        
        # Average line
        if ear_mins:
            avg_ear_min = np.mean(ear_mins)
            ax2.axhline(y=avg_ear_min, color='green', linestyle=':', 
                       linewidth=2, label=f'Average ({avg_ear_min:.3f})', alpha=0.7)
        
        ax2.set_xlabel('Blink ID', fontsize=12)
        ax2.set_ylabel('EAR Minimum Value', fontsize=12)
        ax2.set_title('EAR Minimum Value Trend for Each Blink (Blink Depth)', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.05, 0.30)
        
        # Set x-axis to integers
        if blink_ids:
            ax2.set_xticks(blink_ids)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97], pad=2.0)
        plt.show()


def compare_two_sessions(json_filepath1, json_filepath2):
    """Compare two sessions"""
    print("\n" + "="*60)
    print("Two-Session Comparison Mode")
    print("="*60 + "\n")
    
    print("[Session 1]")
    vis1 = SimpleBlinkVisualizer(json_filepath1)
    print()
    
    print("[Session 2]")
    vis2 = SimpleBlinkVisualizer(json_filepath2)
    print()
    
    blinks1 = vis1.session_data.get('blinks', [])
    blinks2 = vis2.session_data.get('blinks', [])
    
    if not blinks1 or not blinks2:
        print("❌ Error: No blink data available")
        return
    
    # Collect EAR minimum values
    ear_mins1 = [b.get('statistics', {}).get('ear_min', 0) for b in blinks1]
    ear_mins2 = [b.get('statistics', {}).get('ear_min', 0) for b in blinks2]
    
    blink_ids1 = [b.get('blink_id', i+1) for i, b in enumerate(blinks1)]
    blink_ids2 = [b.get('blink_id', i+1) for i, b in enumerate(blinks2)]
    
    # Create graph
    fig, axes = plt.subplots(2, 1, figsize=(16, 11))
    
    session_id1 = vis1.session_data.get('session_id', 'N/A')
    session_id2 = vis2.session_data.get('session_id', 'N/A')
    
    label1 = vis1.session_data.get('label', 'N/A')
    label2 = vis2.session_data.get('label', 'N/A')
    label_str1 = 'Normal State' if label1 == 0 else 'Drowsy State'
    label_str2 = 'Normal State' if label2 == 0 else 'Drowsy State'
    
    fig.suptitle(f'Two-Session Comparison\nSession 1: {label_str1} vs Session 2: {label_str2}', 
                 fontsize=14, fontweight='bold')
    
    # Graph 1: EAR minimum value comparison (line graph)
    ax1 = axes[0]
    
    ax1.plot(blink_ids1, ear_mins1, color='blue', linewidth=2, 
            marker='o', markersize=8, label=f'{label_str1}', alpha=0.7)
    ax1.plot(blink_ids2, ear_mins2, color='red', linewidth=2, 
            marker='s', markersize=8, label=f'{label_str2}', alpha=0.7)
    
    # Threshold line
    ax1.axhline(y=0.21, color='orange', linestyle='--', 
               linewidth=2, label='Threshold (0.21)', alpha=0.7)
    
    # Average lines
    if ear_mins1:
        avg1 = np.mean(ear_mins1)
        ax1.axhline(y=avg1, color='blue', linestyle=':', 
                   linewidth=1.5, label=f'{label_str1} Avg ({avg1:.3f})', alpha=0.5)
    
    if ear_mins2:
        avg2 = np.mean(ear_mins2)
        ax1.axhline(y=avg2, color='red', linestyle=':', 
                   linewidth=1.5, label=f'{label_str2} Avg ({avg2:.3f})', alpha=0.5)
    
    ax1.set_xlabel('Blink ID', fontsize=12)
    ax1.set_ylabel('EAR Minimum Value', fontsize=12)
    ax1.set_title('EAR Minimum Value Comparison for Each Blink', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.05, 0.30)
    
    # Graph 2: EAR minimum value distribution (box plot)
    ax2 = axes[1]
    
    data = [ear_mins1, ear_mins2]
    labels = [label_str1, label_str2]
    colors = ['lightblue', 'lightcoral']
    
    bp = ax2.boxplot(data, labels=labels, patch_artist=True, 
                     widths=0.6, showmeans=True)
    
    # Add colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Display mean values
    for i, (ear_list, label) in enumerate(zip(data, labels)):
        mean_val = np.mean(ear_list)
        median_val = np.median(ear_list)
        ax2.text(i+1, mean_val, f'Mean: {mean_val:.3f}', 
                ha='center', va='bottom', fontsize=9, color='red')
        ax2.text(i+1, median_val, f'Median: {median_val:.3f}', 
                ha='center', va='top', fontsize=9, color='blue')
    
    # Threshold line
    ax2.axhline(y=0.21, color='orange', linestyle='--', 
               linewidth=2, label='Threshold (0.21)', alpha=0.7)
    
    ax2.set_ylabel('EAR Minimum Value', fontsize=12)
    ax2.set_title('EAR Minimum Value Distribution Comparison', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0.05, 0.30)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97], pad=2.0)
    plt.show()
    
    # Display statistical information
    print("\n" + "="*60)
    print("Statistical Information")
    print("="*60)
    print(f"\n[{label_str1}]")
    print(f"  Blink count: {len(ear_mins1)}")
    print(f"  EAR min average: {np.mean(ear_mins1):.3f}")
    print(f"  EAR min median: {np.median(ear_mins1):.3f}")
    print(f"  EAR min std dev: {np.std(ear_mins1):.3f}")
    print(f"  EAR min range: {np.min(ear_mins1):.3f} - {np.max(ear_mins1):.3f}")
    
    print(f"\n[{label_str2}]")
    print(f"  Blink count: {len(ear_mins2)}")
    print(f"  EAR min average: {np.mean(ear_mins2):.3f}")
    print(f"  EAR min median: {np.median(ear_mins2):.3f}")
    print(f"  EAR min std dev: {np.std(ear_mins2):.3f}")
    print(f"  EAR min range: {np.min(ear_mins2):.3f} - {np.max(ear_mins2):.3f}")
    
    print(f"\n[Difference]")
    diff_mean = np.mean(ear_mins2) - np.mean(ear_mins1)
    print(f"  Mean difference: {diff_mean:+.3f}")
    print(f"  Change rate: {(diff_mean / np.mean(ear_mins1) * 100):+.1f}%")
    print("="*60 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simple Blink EAR Visualizer')
    parser.add_argument('json_files', nargs='+', help='JSON file path(s) (one or two)')
    
    args = parser.parse_args()
    
    # matplotlib settings
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.autolayout'] = False
    except:
        pass
    
    if len(args.json_files) == 1:
        # Single file visualization
        print("\n" + "="*60)
        print("Single Session Display Mode")
        print("="*60 + "\n")
        
        visualizer = SimpleBlinkVisualizer(args.json_files[0])
        print()
        visualizer.plot_session_ear_timeline()
    
    elif len(args.json_files) == 2:
        # Two file comparison
        compare_two_sessions(args.json_files[0], args.json_files[1])
    
    else:
        print("❌ Error: Please specify one or two JSON files")
        print("Usage:")
        print("  1 file: python simple_blink_visualizer.py file1.json")
        print("  2 files: python simple_blink_visualizer.py file1.json file2.json")


if __name__ == "__main__":
    main()