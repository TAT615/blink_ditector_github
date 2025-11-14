"""
眠気検出システム - 瞬き時系列データ可視化プログラム

JSONファイルから時系列の瞬きの動きをグラフ表示します。
- EARの時系列グラフ
- 上まぶた円半径の時系列
- 下まぶた円半径の時系列
- 垂直距離の時系列
- 瞬き状態の色分け表示

使い方:
    # 1つのJSONファイルを表示
    python visualize_blink_timeseries.py data/sessions/20251112_143000_001_normal.json
    
    # 複数のJSONファイルを比較
    python visualize_blink_timeseries.py data/sessions/20251112_143000_001_normal.json data/sessions/20251112_150000_001_drowsy.json
    
    # 特定の瞬きだけを表示
    python visualize_blink_timeseries.py data/sessions/20251112_143000_001_normal.json --blink-ids 1 2 3
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path


class BlinkVisualizer:
    """瞬き時系列データの可視化クラス"""
    
    # 瞬き状態の色定義
    STATE_COLORS = {
        'OPEN': '#2ecc71',      # 緑
        'CLOSING': '#f39c12',   # オレンジ
        'CLOSED': '#e74c3c',    # 赤
        'OPENING': '#3498db'    # 青
    }
    
    def __init__(self, json_filepath):
        self.json_filepath = json_filepath
        self.session_data = None
        self.load_data()
    
    def load_data(self):
        """JSONファイルを読み込み"""
        try:
            with open(self.json_filepath, 'r', encoding='utf-8') as f:
                self.session_data = json.load(f)
            
            print(f"✓ ファイル読み込み成功: {self.json_filepath}")
            print(f"  セッションID: {self.session_data.get('session_id', 'N/A')}")
            print(f"  ユーザーID: {self.session_data.get('user_id', 'N/A')}")
            print(f"  ラベル: {self.session_data.get('label', 'N/A')} ({'正常' if self.session_data.get('label') == 0 else '眠気'})")
            print(f"  総瞬き数: {self.session_data.get('total_blinks', 0)}")
            print()
            
        except FileNotFoundError:
            print(f"❌ エラー: ファイルが見つかりません: {self.json_filepath}")
            raise
        except json.JSONDecodeError:
            print(f"❌ エラー: JSONファイルの形式が不正です: {self.json_filepath}")
            raise
    
    def plot_single_blink(self, blink_id):
        """1つの瞬きの時系列データを表示"""
        blinks = self.session_data.get('blinks', [])
        
        if blink_id < 1 or blink_id > len(blinks):
            print(f"❌ エラー: 瞬きID {blink_id} は存在しません（範囲: 1-{len(blinks)}）")
            return
        
        blink = blinks[blink_id - 1]
        
        # EAR時系列データ
        ear_timeseries = blink.get('ear_timeseries', [])
        
        # 2円時系列データ
        circles_timeseries = blink.get('circles_timeseries', [])
        
        if not ear_timeseries or not circles_timeseries:
            print(f"⚠️  警告: 瞬きID {blink_id} の時系列データが存在しません")
            return
        
        # 統計情報
        stats = blink.get('statistics', {})
        
        # グラフ作成
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        fig.suptitle(f"瞬き #{blink_id} の時系列データ\n{self.session_data.get('session_id', 'N/A')}", 
                     fontsize=14, fontweight='bold')
        
        # 時間軸の準備（最初の時刻を0とする）
        ear_times = [p['timestamp'] - ear_timeseries[0]['timestamp'] for p in ear_timeseries]
        ear_values = [p['ear'] for p in ear_timeseries]
        ear_states = [p['state'] for p in ear_timeseries]
        
        circles_times = [p['timestamp'] - circles_timeseries[0]['timestamp'] for p in circles_timeseries]
        upper_radii = [p['upper_radius'] for p in circles_timeseries]
        lower_radii = [p['lower_radius'] for p in circles_timeseries]
        vert_distances = [p['vertical_distance'] for p in circles_timeseries]
        
        # 1. EARのグラフ
        ax1 = axes[0]
        self._plot_timeseries_with_states(ax1, ear_times, ear_values, ear_states, 
                                          'EAR (Eye Aspect Ratio)', 'EAR値')
        ax1.axhline(y=0.21, color='red', linestyle='--', linewidth=1, alpha=0.5, label='閾値 (0.21)')
        ax1.legend(loc='upper right')
        
        # 統計情報を表示
        info_text = (f"閉眼時間: {stats.get('closing_time', 0)*1000:.0f}ms\n"
                    f"開眼時間: {stats.get('opening_time', 0)*1000:.0f}ms\n"
                    f"瞬き係数: {stats.get('blink_coefficient', 0):.2f}")
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
        
        # 2. 上まぶた円半径のグラフ
        ax2 = axes[1]
        self._plot_timeseries_simple(ax2, circles_times, upper_radii, 
                                     '上まぶた円半径（C1）', '半径 (px)', 'red')
        
        # 3. 下まぶた円半径のグラフ
        ax3 = axes[2]
        self._plot_timeseries_simple(ax3, circles_times, lower_radii, 
                                     '下まぶた円半径（C2）', '半径 (px)', 'blue')
        
        # 4. 垂直距離のグラフ
        ax4 = axes[3]
        self._plot_timeseries_simple(ax4, circles_times, vert_distances, 
                                     '垂直距離（まぶた間）', '距離 (px)', 'green')
        
        # 統計情報を表示
        info_text2 = (f"上半径最大: {stats.get('upper_radius_max', 0):.1f}px\n"
                     f"下半径最大: {stats.get('lower_radius_max', 0):.1f}px\n"
                     f"垂直距離最小: {stats.get('vertical_distance_min', 0):.1f}px")
        ax4.text(0.02, 0.98, info_text2, transform=ax4.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_multiple_blinks(self, blink_ids=None, max_blinks=5):
        """複数の瞬きを並べて表示"""
        blinks = self.session_data.get('blinks', [])
        
        if not blinks:
            print("❌ エラー: 瞬きデータがありません")
            return
        
        # 表示する瞬きIDを決定
        if blink_ids is None:
            # 指定がない場合は最初のmax_blinks個
            blink_ids = list(range(1, min(max_blinks + 1, len(blinks) + 1)))
        else:
            # 範囲チェック
            blink_ids = [bid for bid in blink_ids if 1 <= bid <= len(blinks)]
        
        if not blink_ids:
            print("❌ エラー: 有効な瞬きIDがありません")
            return
        
        num_blinks = len(blink_ids)
        
        # グラフ作成
        fig, axes = plt.subplots(num_blinks, 2, figsize=(14, 4 * num_blinks))
        
        if num_blinks == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f"複数瞬きの比較\n{self.session_data.get('session_id', 'N/A')}", 
                     fontsize=14, fontweight='bold')
        
        for i, blink_id in enumerate(blink_ids):
            blink = blinks[blink_id - 1]
            
            # EAR時系列
            ear_timeseries = blink.get('ear_timeseries', [])
            if not ear_timeseries:
                continue
            
            # 2円時系列
            circles_timeseries = blink.get('circles_timeseries', [])
            if not circles_timeseries:
                continue
            
            # 統計情報
            stats = blink.get('statistics', {})
            
            # 時間軸の準備
            ear_times = [p['timestamp'] - ear_timeseries[0]['timestamp'] for p in ear_timeseries]
            ear_values = [p['ear'] for p in ear_timeseries]
            ear_states = [p['state'] for p in ear_timeseries]
            
            circles_times = [p['timestamp'] - circles_timeseries[0]['timestamp'] for p in circles_timeseries]
            upper_radii = [p['upper_radius'] for p in circles_timeseries]
            lower_radii = [p['lower_radius'] for p in circles_timeseries]
            vert_distances = [p['vertical_distance'] for p in circles_timeseries]
            
            # 左側: EAR
            ax_left = axes[i, 0]
            self._plot_timeseries_with_states(ax_left, ear_times, ear_values, ear_states, 
                                              f'瞬き #{blink_id}: EAR', 'EAR値')
            ax_left.axhline(y=0.21, color='red', linestyle='--', linewidth=1, alpha=0.5)
            
            # 統計情報
            info_text = (f"Tc: {stats.get('closing_time', 0)*1000:.0f}ms, "
                        f"To: {stats.get('opening_time', 0)*1000:.0f}ms, "
                        f"係数: {stats.get('blink_coefficient', 0):.2f}")
            ax_left.text(0.5, 0.02, info_text, transform=ax_left.transAxes,
                        horizontalalignment='center', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 右側: 2円パラメータ
            ax_right = axes[i, 1]
            ax_right.plot(circles_times, upper_radii, 'r-', linewidth=2, label='上まぶた円', marker='o')
            ax_right.plot(circles_times, lower_radii, 'b-', linewidth=2, label='下まぶた円', marker='s')
            ax_right.plot(circles_times, vert_distances, 'g-', linewidth=2, label='垂直距離', marker='^')
            ax_right.set_xlabel('時間 (秒)')
            ax_right.set_ylabel('値 (px)')
            ax_right.set_title(f'瞬き #{blink_id}: 2円パラメータ')
            ax_right.legend(loc='best')
            ax_right.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_blinks_overview(self):
        """全瞬きの概要を表示"""
        blinks = self.session_data.get('blinks', [])
        
        if not blinks:
            print("❌ エラー: 瞬きデータがありません")
            return
        
        # 統計量を収集
        blink_ids = []
        closing_times = []
        opening_times = []
        blink_coefficients = []
        upper_radii = []
        lower_radii = []
        vert_distances = []
        
        for blink in blinks:
            stats = blink.get('statistics', {})
            blink_ids.append(blink.get('blink_id', 0))
            closing_times.append(stats.get('closing_time', 0) * 1000)  # ms
            opening_times.append(stats.get('opening_time', 0) * 1000)  # ms
            blink_coefficients.append(stats.get('blink_coefficient', 0))
            upper_radii.append(stats.get('upper_radius_max', 0))
            lower_radii.append(stats.get('lower_radius_max', 0))
            vert_distances.append(stats.get('vertical_distance_min', 0))
        
        # グラフ作成
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"全瞬きの統計量概要\n{self.session_data.get('session_id', 'N/A')}", 
                     fontsize=14, fontweight='bold')
        
        # 1. 閉眼時間
        ax1 = axes[0, 0]
        ax1.bar(blink_ids, closing_times, color='orange', alpha=0.7)
        ax1.set_xlabel('瞬きID')
        ax1.set_ylabel('時間 (ms)')
        ax1.set_title('閉眼時間 (Tc)')
        ax1.axhline(y=np.mean(closing_times), color='red', linestyle='--', 
                   label=f'平均: {np.mean(closing_times):.1f}ms')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 開眼時間
        ax2 = axes[0, 1]
        ax2.bar(blink_ids, opening_times, color='blue', alpha=0.7)
        ax2.set_xlabel('瞬きID')
        ax2.set_ylabel('時間 (ms)')
        ax2.set_title('開眼時間 (To)')
        ax2.axhline(y=np.mean(opening_times), color='red', linestyle='--', 
                   label=f'平均: {np.mean(opening_times):.1f}ms')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 瞬き係数
        ax3 = axes[0, 2]
        ax3.bar(blink_ids, blink_coefficients, color='green', alpha=0.7)
        ax3.set_xlabel('瞬きID')
        ax3.set_ylabel('係数')
        ax3.set_title('瞬き係数 (To/Tc)')
        ax3.axhline(y=np.mean(blink_coefficients), color='red', linestyle='--', 
                   label=f'平均: {np.mean(blink_coefficients):.2f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 上まぶた円半径
        ax4 = axes[1, 0]
        ax4.bar(blink_ids, upper_radii, color='red', alpha=0.7)
        ax4.set_xlabel('瞬きID')
        ax4.set_ylabel('半径 (px)')
        ax4.set_title('上まぶた円半径最大')
        ax4.axhline(y=np.mean(upper_radii), color='darkred', linestyle='--', 
                   label=f'平均: {np.mean(upper_radii):.1f}px')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 下まぶた円半径
        ax5 = axes[1, 1]
        ax5.bar(blink_ids, lower_radii, color='blue', alpha=0.7)
        ax5.set_xlabel('瞬きID')
        ax5.set_ylabel('半径 (px)')
        ax5.set_title('下まぶた円半径最大')
        ax5.axhline(y=np.mean(lower_radii), color='darkblue', linestyle='--', 
                   label=f'平均: {np.mean(lower_radii):.1f}px')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 垂直距離
        ax6 = axes[1, 2]
        ax6.bar(blink_ids, vert_distances, color='purple', alpha=0.7)
        ax6.set_xlabel('瞬きID')
        ax6.set_ylabel('距離 (px)')
        ax6.set_title('垂直距離最小')
        ax6.axhline(y=np.mean(vert_distances), color='darkviolet', linestyle='--', 
                   label=f'平均: {np.mean(vert_distances):.1f}px')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_timeseries_with_states(self, ax, times, values, states, title, ylabel):
        """状態に応じて色分けして時系列データをプロット"""
        # 状態ごとにグループ化
        for i in range(len(times)):
            state = states[i]
            color = self.STATE_COLORS.get(state, 'gray')
            
            if i < len(times) - 1:
                ax.plot(times[i:i+2], values[i:i+2], color=color, linewidth=2)
                ax.scatter(times[i], values[i], color=color, s=30, zorder=5)
            else:
                ax.scatter(times[i], values[i], color=color, s=30, zorder=5)
        
        # 凡例用のダミープロット
        for state, color in self.STATE_COLORS.items():
            ax.plot([], [], color=color, linewidth=2, label=state)
        
        ax.set_xlabel('時間 (秒)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    def _plot_timeseries_simple(self, ax, times, values, title, ylabel, color):
        """シンプルな時系列データをプロット"""
        ax.plot(times, values, color=color, linewidth=2, marker='o')
        ax.set_xlabel('時間 (秒)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)


def compare_two_sessions(json_filepath1, json_filepath2):
    """2つのセッションを比較"""
    vis1 = BlinkVisualizer(json_filepath1)
    vis2 = BlinkVisualizer(json_filepath2)
    
    # 最初の瞬きを比較
    blinks1 = vis1.session_data.get('blinks', [])
    blinks2 = vis2.session_data.get('blinks', [])
    
    if not blinks1 or not blinks2:
        print("❌ エラー: 瞬きデータがありません")
        return
    
    blink1 = blinks1[0]
    blink2 = blinks2[0]
    
    # グラフ作成
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('2つのセッションの比較（瞬き#1）', fontsize=14, fontweight='bold')
    
    # セッション1
    ear_timeseries1 = blink1.get('ear_timeseries', [])
    ear_times1 = [p['timestamp'] - ear_timeseries1[0]['timestamp'] for p in ear_timeseries1]
    ear_values1 = [p['ear'] for p in ear_timeseries1]
    
    circles_timeseries1 = blink1.get('circles_timeseries', [])
    circles_times1 = [p['timestamp'] - circles_timeseries1[0]['timestamp'] for p in circles_timeseries1]
    upper_radii1 = [p['upper_radius'] for p in circles_timeseries1]
    vert_distances1 = [p['vertical_distance'] for p in circles_timeseries1]
    
    # セッション2
    ear_timeseries2 = blink2.get('ear_timeseries', [])
    ear_times2 = [p['timestamp'] - ear_timeseries2[0]['timestamp'] for p in ear_timeseries2]
    ear_values2 = [p['ear'] for p in ear_timeseries2]
    
    circles_timeseries2 = blink2.get('circles_timeseries', [])
    circles_times2 = [p['timestamp'] - circles_timeseries2[0]['timestamp'] for p in circles_timeseries2]
    upper_radii2 = [p['upper_radius'] for p in circles_timeseries2]
    vert_distances2 = [p['vertical_distance'] for p in circles_timeseries2]
    
    # EAR比較
    ax1 = axes[0, 0]
    ax1.plot(ear_times1, ear_values1, 'b-', linewidth=2, label=vis1.session_data.get('session_id'), marker='o')
    ax1.plot(ear_times2, ear_values2, 'r-', linewidth=2, label=vis2.session_data.get('session_id'), marker='s')
    ax1.axhline(y=0.21, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('時間 (秒)')
    ax1.set_ylabel('EAR値')
    ax1.set_title('EAR比較')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 上まぶた円半径比較
    ax2 = axes[0, 1]
    ax2.plot(circles_times1, upper_radii1, 'b-', linewidth=2, label=vis1.session_data.get('session_id'), marker='o')
    ax2.plot(circles_times2, upper_radii2, 'r-', linewidth=2, label=vis2.session_data.get('session_id'), marker='s')
    ax2.set_xlabel('時間 (秒)')
    ax2.set_ylabel('半径 (px)')
    ax2.set_title('上まぶた円半径比較')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 垂直距離比較
    ax3 = axes[1, 0]
    ax3.plot(circles_times1, vert_distances1, 'b-', linewidth=2, label=vis1.session_data.get('session_id'), marker='o')
    ax3.plot(circles_times2, vert_distances2, 'r-', linewidth=2, label=vis2.session_data.get('session_id'), marker='s')
    ax3.set_xlabel('時間 (秒)')
    ax3.set_ylabel('距離 (px)')
    ax3.set_title('垂直距離比較')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 統計比較
    ax4 = axes[1, 1]
    stats1 = blink1.get('statistics', {})
    stats2 = blink2.get('statistics', {})
    
    categories = ['閉眼時間\n(ms)', '開眼時間\n(ms)', '瞬き係数', '垂直距離最小\n(px)']
    values1 = [
        stats1.get('closing_time', 0) * 1000,
        stats1.get('opening_time', 0) * 1000,
        stats1.get('blink_coefficient', 0),
        stats1.get('vertical_distance_min', 0)
    ]
    values2 = [
        stats2.get('closing_time', 0) * 1000,
        stats2.get('opening_time', 0) * 1000,
        stats2.get('blink_coefficient', 0),
        stats2.get('vertical_distance_min', 0)
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # 正規化（視覚化のため）
    max_vals = [max(values1[i], values2[i]) for i in range(len(values1))]
    norm_values1 = [values1[i] / max_vals[i] if max_vals[i] > 0 else 0 for i in range(len(values1))]
    norm_values2 = [values2[i] / max_vals[i] if max_vals[i] > 0 else 0 for i in range(len(values2))]
    
    ax4.bar(x - width/2, norm_values1, width, label=vis1.session_data.get('session_id'), alpha=0.8)
    ax4.bar(x + width/2, norm_values2, width, label=vis2.session_data.get('session_id'), alpha=0.8)
    ax4.set_ylabel('正規化された値')
    ax4.set_title('統計量比較（正規化）')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=9)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 実際の値をテキストで表示
    for i, (v1, v2) in enumerate(zip(values1, values2)):
        ax4.text(i - width/2, norm_values1[i] + 0.05, f'{v1:.1f}', 
                ha='center', va='bottom', fontsize=8)
        ax4.text(i + width/2, norm_values2[i] + 0.05, f'{v2:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='瞬き時系列データ可視化プログラム')
    parser.add_argument('json_files', nargs='+', help='JSONファイルのパス（複数指定可能）')
    parser.add_argument('--blink-ids', nargs='+', type=int, 
                       help='表示する瞬きIDを指定（例: --blink-ids 1 2 3）')
    parser.add_argument('--mode', choices=['single', 'multiple', 'overview', 'compare'], 
                       default='overview',
                       help='表示モード: single=1瞬き詳細, multiple=複数瞬き比較, overview=全瞬き概要, compare=2セッション比較')
    
    args = parser.parse_args()
    
    # matplotlibの日本語フォント設定（オプション）
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    except:
        pass
    
    if args.mode == 'compare' and len(args.json_files) == 2:
        # 2セッション比較
        print("モード: 2セッション比較\n")
        compare_two_sessions(args.json_files[0], args.json_files[1])
    
    elif len(args.json_files) == 1:
        # 1ファイルの可視化
        visualizer = BlinkVisualizer(args.json_files[0])
        
        if args.mode == 'single':
            # 1つの瞬きの詳細
            blink_id = args.blink_ids[0] if args.blink_ids else 1
            print(f"モード: 単一瞬き詳細（瞬きID: {blink_id}）\n")
            visualizer.plot_single_blink(blink_id)
        
        elif args.mode == 'multiple':
            # 複数の瞬きを並べて表示
            print(f"モード: 複数瞬き比較\n")
            visualizer.plot_multiple_blinks(blink_ids=args.blink_ids)
        
        elif args.mode == 'overview':
            # 全瞬きの概要
            print(f"モード: 全瞬き概要\n")
            visualizer.plot_all_blinks_overview()
    
    else:
        print("❌ エラー: 複数ファイルの可視化は compare モードのみ対応しています")
        print("使い方: python visualize_blink_timeseries.py file1.json file2.json --mode compare")


if __name__ == "__main__":
    main()
