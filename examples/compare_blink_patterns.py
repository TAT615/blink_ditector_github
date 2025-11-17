"""
眠気検出システム - 瞬き比較可視化プログラム

通常時と眠気時の瞬きデータを比較して、詳細な時系列グラフを生成します。

使用方法:
    python compare_blink_patterns.py normal.json drowsy.json
    python compare_blink_patterns.py normal.json drowsy.json --blink-id 5
    python compare_blink_patterns.py normal.json drowsy.json --num-blinks 3
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import argparse


class BlinkPatternComparator:
    """通常時と眠気時の瞬きパターンを比較"""
    
    # 瞬き状態ごとの色
    STATE_COLORS = {
        'OPEN': '#90EE90',      # ライトグリーン
        'CLOSING': '#FFD700',   # ゴールド
        'CLOSED': '#FF6B6B',    # ライトレッド
        'OPENING': '#87CEEB'    # スカイブルー
    }
    
    def __init__(self, normal_json_path, drowsy_json_path):
        self.normal_json_path = normal_json_path
        self.drowsy_json_path = drowsy_json_path
        self.normal_data = None
        self.drowsy_data = None
        self.load_data()
    
    def load_data(self):
        """JSONファイルを読み込み"""
        with open(self.normal_json_path, 'r', encoding='utf-8') as f:
            self.normal_data = json.load(f)
        
        with open(self.drowsy_json_path, 'r', encoding='utf-8') as f:
            self.drowsy_data = json.load(f)
        
        print("=" * 80)
        print("データ読み込み完了")
        print("=" * 80)
        print(f"\n【通常時データ】")
        print(f"  セッションID: {self.normal_data['session_id']}")
        print(f"  KSS Score: {self.normal_data['kss_score']}")
        print(f"  有効な瞬き数: {self.normal_data['valid_blinks']}")
        
        print(f"\n【眠気時データ】")
        print(f"  セッションID: {self.drowsy_data['session_id']}")
        print(f"  KSS Score: {self.drowsy_data['kss_score']}")
        print(f"  有効な瞬き数: {self.drowsy_data['valid_blinks']}")
        print()
    
    def compare_single_blink(self, normal_blink_id=None, drowsy_blink_id=None, save_path=None):
        """
        単一の瞬きを詳細比較
        
        Args:
            normal_blink_id: 通常時の瞬きID（Noneの場合は中央付近を選択）
            drowsy_blink_id: 眠気時の瞬きID（Noneの場合は中央付近を選択）
            save_path: 保存先パス
        """
        normal_blinks = self.normal_data['blinks']
        drowsy_blinks = self.drowsy_data['blinks']
        
        # デフォルトでは中央付近の瞬きを選択
        if normal_blink_id is None:
            normal_blink_id = len(normal_blinks) // 2
        if drowsy_blink_id is None:
            drowsy_blink_id = len(drowsy_blinks) // 2
        
        # インデックスに変換（1-basedから0-basedへ）
        normal_idx = normal_blink_id - 1 if normal_blink_id > 0 else normal_blink_id
        drowsy_idx = drowsy_blink_id - 1 if drowsy_blink_id > 0 else drowsy_blink_id
        
        normal_blink = normal_blinks[normal_idx]
        drowsy_blink = drowsy_blinks[drowsy_idx]
        
        # グラフ作成
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)
        
        fig.suptitle(f'Blink Pattern Comparison: Normal (KSS={self.normal_data["kss_score"]}) vs Drowsy (KSS={self.drowsy_data["kss_score"]})',
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. EAR比較（左右並べて）
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_ear_timeseries(ax1, normal_blink, 'Normal', 'blue')
        self._plot_ear_timeseries(ax2, drowsy_blink, 'Drowsy', 'red')
        
        # 2. 上まぶた円半径比較
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_upper_radius(ax3, normal_blink, 'Normal', 'blue')
        self._plot_upper_radius(ax4, drowsy_blink, 'Drowsy', 'red')
        
        # 3. 下まぶた円半径比較
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_lower_radius(ax5, normal_blink, 'Normal', 'blue')
        self._plot_lower_radius(ax6, drowsy_blink, 'Drowsy', 'red')
        
        # 4. 垂直距離比較
        ax7 = fig.add_subplot(gs[3, 0])
        ax8 = fig.add_subplot(gs[3, 1])
        self._plot_vertical_distance(ax7, normal_blink, 'Normal', 'blue')
        self._plot_vertical_distance(ax8, drowsy_blink, 'Drowsy', 'red')
        
        # 統計情報の追加
        self._add_statistics_text(fig, normal_blink, drowsy_blink)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ グラフを保存しました: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_multiple_blinks(self, num_blinks=3, save_path=None):
        """
        複数の瞬きを重ね合わせて比較
        
        Args:
            num_blinks: 比較する瞬きの数
            save_path: 保存先パス
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Multiple Blink Patterns Overlay: Normal vs Drowsy (n={num_blinks} each)',
                     fontsize=16, fontweight='bold')
        
        # 通常時と眠気時からランダムに瞬きを選択
        normal_blinks = self.normal_data['blinks']
        drowsy_blinks = self.drowsy_data['blinks']
        
        # 均等に分散して選択
        normal_indices = np.linspace(5, len(normal_blinks)-5, num_blinks, dtype=int)
        drowsy_indices = np.linspace(5, len(drowsy_blinks)-5, num_blinks, dtype=int)
        
        # EARの重ね合わせ
        ax = axes[0, 0]
        for idx in normal_indices:
            self._plot_ear_overlay(ax, normal_blinks[idx], 'blue', alpha=0.3)
        ax.set_title('Normal - EAR Overlay', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('EAR', fontweight='bold')
        ax.axhline(y=0.21, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(alpha=0.3)
        
        ax = axes[0, 1]
        for idx in drowsy_indices:
            self._plot_ear_overlay(ax, drowsy_blinks[idx], 'red', alpha=0.3)
        ax.set_title('Drowsy - EAR Overlay', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('EAR', fontweight='bold')
        ax.axhline(y=0.21, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(alpha=0.3)
        
        # 垂直距離の重ね合わせ
        ax = axes[1, 0]
        for idx in normal_indices:
            self._plot_vertical_distance_overlay(ax, normal_blinks[idx], 'blue', alpha=0.3)
        ax.set_title('Normal - Vertical Distance Overlay', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Vertical Distance (px)', fontweight='bold')
        ax.grid(alpha=0.3)
        
        ax = axes[1, 1]
        for idx in drowsy_indices:
            self._plot_vertical_distance_overlay(ax, drowsy_blinks[idx], 'red', alpha=0.3)
        ax.set_title('Drowsy - Vertical Distance Overlay', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Vertical Distance (px)', fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ グラフを保存しました: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_ear_timeseries(self, ax, blink, label, color):
        """EARの時系列をプロット（状態別に色分け）"""
        ear_ts = blink['ear_timeseries']
        stats = blink['statistics']
        
        if not ear_ts:
            return
        
        # 時間を0から開始
        base_time = ear_ts[0]['timestamp']
        times = [(p['timestamp'] - base_time) * 1000 for p in ear_ts]  # ミリ秒に変換
        ear_values = [p['ear'] for p in ear_ts]
        states = [p['state'] for p in ear_ts]
        
        # 状態ごとに色分けしてプロット
        for i in range(len(times) - 1):
            ax.plot([times[i], times[i+1]], [ear_values[i], ear_values[i+1]], 
                   color=self.STATE_COLORS.get(states[i], 'gray'),
                   linewidth=2.5, marker='o', markersize=4)
        
        # 最後の点
        if len(times) > 0:
            ax.plot(times[-1], ear_values[-1], 
                   color=self.STATE_COLORS.get(states[-1], 'gray'),
                   marker='o', markersize=4)
        
        ax.axhline(y=0.21, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
        ax.set_title(f'{label} - EAR Time Series', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (ms)', fontweight='bold')
        ax.set_ylabel('EAR', fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim([0, 0.35])
        
        # 統計情報を表示
        info_text = (f"Closing: {stats['closing_time']*1000:.1f}ms\n"
                    f"Opening: {stats['opening_time']*1000:.1f}ms\n"
                    f"Total: {stats['total_duration']*1000:.1f}ms\n"
                    f"Coefficient: {stats['blink_coefficient']:.2f}")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # 凡例を追加
        legend_elements = [mpatches.Patch(facecolor=self.STATE_COLORS[state], 
                                         label=state) 
                          for state in ['CLOSING', 'CLOSED', 'OPENING']]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    def _plot_upper_radius(self, ax, blink, label, color):
        """上まぶた円半径をプロット"""
        circles_ts = blink['circles_timeseries']
        
        if not circles_ts:
            return
        
        base_time = circles_ts[0]['timestamp']
        times = [(p['timestamp'] - base_time) * 1000 for p in circles_ts]
        radii = [p['upper_radius'] for p in circles_ts]
        states = [p['state'] for p in circles_ts]
        
        # 状態ごとに色分け
        for i in range(len(times) - 1):
            ax.plot([times[i], times[i+1]], [radii[i], radii[i+1]], 
                   color=self.STATE_COLORS.get(states[i], 'gray'),
                   linewidth=2.5, marker='o', markersize=4)
        
        if len(times) > 0:
            ax.plot(times[-1], radii[-1], 
                   color=self.STATE_COLORS.get(states[-1], 'gray'),
                   marker='o', markersize=4)
        
        ax.set_title(f'{label} - Upper Eyelid Radius (C1)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (ms)', fontweight='bold')
        ax.set_ylabel('Radius (px)', fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
    
    def _plot_lower_radius(self, ax, blink, label, color):
        """下まぶた円半径をプロット"""
        circles_ts = blink['circles_timeseries']
        
        if not circles_ts:
            return
        
        base_time = circles_ts[0]['timestamp']
        times = [(p['timestamp'] - base_time) * 1000 for p in circles_ts]
        radii = [p['lower_radius'] for p in circles_ts]
        states = [p['state'] for p in circles_ts]
        
        for i in range(len(times) - 1):
            ax.plot([times[i], times[i+1]], [radii[i], radii[i+1]], 
                   color=self.STATE_COLORS.get(states[i], 'gray'),
                   linewidth=2.5, marker='o', markersize=4)
        
        if len(times) > 0:
            ax.plot(times[-1], radii[-1], 
                   color=self.STATE_COLORS.get(states[-1], 'gray'),
                   marker='o', markersize=4)
        
        ax.set_title(f'{label} - Lower Eyelid Radius (C2)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (ms)', fontweight='bold')
        ax.set_ylabel('Radius (px)', fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
    
    def _plot_vertical_distance(self, ax, blink, label, color):
        """垂直距離をプロット"""
        circles_ts = blink['circles_timeseries']
        
        if not circles_ts:
            return
        
        base_time = circles_ts[0]['timestamp']
        times = [(p['timestamp'] - base_time) * 1000 for p in circles_ts]
        distances = [p['vertical_distance'] for p in circles_ts]
        states = [p['state'] for p in circles_ts]
        
        for i in range(len(times) - 1):
            ax.plot([times[i], times[i+1]], [distances[i], distances[i+1]], 
                   color=self.STATE_COLORS.get(states[i], 'gray'),
                   linewidth=2.5, marker='o', markersize=4)
        
        if len(times) > 0:
            ax.plot(times[-1], distances[-1], 
                   color=self.STATE_COLORS.get(states[-1], 'gray'),
                   marker='o', markersize=4)
        
        ax.set_title(f'{label} - Vertical Distance (Eye Opening)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (ms)', fontweight='bold')
        ax.set_ylabel('Distance (px)', fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
    
    def _plot_ear_overlay(self, ax, blink, color, alpha=0.5):
        """EARを重ね合わせてプロット（正規化された時間軸）"""
        ear_ts = blink['ear_timeseries']
        
        if not ear_ts:
            return
        
        # 0-1に正規化された時間軸
        times = np.linspace(0, 1, len(ear_ts))
        ear_values = [p['ear'] for p in ear_ts]
        
        ax.plot(times, ear_values, color=color, alpha=alpha, linewidth=1.5)
    
    def _plot_vertical_distance_overlay(self, ax, blink, color, alpha=0.5):
        """垂直距離を重ね合わせてプロット（正規化された時間軸）"""
        circles_ts = blink['circles_timeseries']
        
        if not circles_ts:
            return
        
        times = np.linspace(0, 1, len(circles_ts))
        distances = [p['vertical_distance'] for p in circles_ts]
        
        ax.plot(times, distances, color=color, alpha=alpha, linewidth=1.5)
    
    def _add_statistics_text(self, fig, normal_blink, drowsy_blink):
        """統計情報をテキストで追加"""
        normal_stats = normal_blink['statistics']
        drowsy_stats = drowsy_blink['statistics']
        
        # 比較テキストを作成
        comparison_text = "📊 Comparison Summary\n" + "="*40 + "\n"
        
        params = [
            ('Closing Time', 'closing_time', 'ms'),
            ('Opening Time', 'opening_time', 'ms'),
            ('Total Duration', 'total_duration', 'ms'),
            ('Blink Coefficient', 'blink_coefficient', '')
        ]
        
        for param_name, param_key, unit in params:
            normal_val = normal_stats[param_key]
            drowsy_val = drowsy_stats[param_key]
            
            if unit == 'ms':
                normal_val *= 1000
                drowsy_val *= 1000
            
            diff = drowsy_val - normal_val
            percent = (diff / normal_val * 100) if normal_val != 0 else 0
            
            comparison_text += f"{param_name}:\n"
            comparison_text += f"  Normal: {normal_val:.1f}{unit}\n"
            comparison_text += f"  Drowsy: {drowsy_val:.1f}{unit}\n"
            comparison_text += f"  Diff: {diff:+.1f}{unit} ({percent:+.1f}%)\n\n"
        
        # 図の下部に配置
        fig.text(0.5, 0.02, comparison_text, ha='center', va='bottom',
                fontfamily='monospace', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))


def main():
    parser = argparse.ArgumentParser(description='通常時と眠気時の瞬きパターンを比較')
    parser.add_argument('normal_json', help='通常時のJSONファイルパス')
    parser.add_argument('drowsy_json', help='眠気時のJSONファイルパス')
    parser.add_argument('--blink-id', type=int, default=None,
                       help='比較する特定の瞬きID（両方に同じIDを使用）')
    parser.add_argument('--normal-blink-id', type=int, default=None,
                       help='通常時の瞬きID')
    parser.add_argument('--drowsy-blink-id', type=int, default=None,
                       help='眠気時の瞬きID')
    parser.add_argument('--num-blinks', type=int, default=3,
                       help='重ね合わせ比較する瞬きの数')
    parser.add_argument('--output-dir', default='graph/outputs',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 比較オブジェクトの作成
    comparator = BlinkPatternComparator(args.normal_json, args.drowsy_json)
    
    # 単一瞬きの比較
    normal_id = args.normal_blink_id if args.normal_blink_id else args.blink_id
    drowsy_id = args.drowsy_blink_id if args.drowsy_blink_id else args.blink_id
    
    print("\n" + "="*80)
    print("単一瞬きの詳細比較を生成中...")
    print("="*80)
    save_path1 = output_dir / 'blink_comparison_single.png'
    comparator.compare_single_blink(normal_id, drowsy_id, save_path1)
    
    # 複数瞬きの重ね合わせ比較
    print("\n" + "="*80)
    print(f"複数瞬き（n={args.num_blinks}）の重ね合わせ比較を生成中...")
    print("="*80)
    save_path2 = output_dir / 'blink_comparison_overlay.png'
    comparator.compare_multiple_blinks(args.num_blinks, save_path2)
    
    print("\n" + "="*80)
    print("✓ すべてのグラフ生成が完了しました！")
    print("="*80)
    print(f"\n生成されたファイル:")
    print(f"  1. {save_path1}")
    print(f"  2. {save_path2}")
    print()


if __name__ == '__main__':
    main()
