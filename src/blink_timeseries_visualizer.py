"""
時系列データの可視化ツール

EARと楕円パラメータの時系列データをグラフ化します。
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


class BlinkTimeSeriesVisualizer:
    """
    瞬きの時系列データを可視化
    """
    
    def __init__(self):
        # 日本語フォントの設定（環境に応じて調整）
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 状態ごとの色
        self.state_colors = {
            'OPEN': '#90EE90',      # ライトグリーン
            'CLOSING': '#FFD700',   # ゴールド
            'CLOSED': '#FF6B6B',    # ライトレッド
            'OPENING': '#87CEEB'    # スカイブルー
        }
    
    def visualize_single_blink(self, blink_data, save_path=None):
        """
        1つの瞬きの時系列データを可視化
        
        Args:
            blink_data (dict): 瞬きデータ
            save_path (str): 保存先パス（Noneの場合は表示のみ）
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Blink {blink_data["blink_id"]} - Time Series Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 時系列データの取得
        ear_ts = blink_data['ear_timeseries']
        ellipse_ts = blink_data['ellipse_timeseries']
        
        if len(ear_ts) == 0 or len(ellipse_ts) == 0:
            print("警告: 時系列データが空です")
            return
        
        # 基準時刻からの相対時間に変換
        base_time = blink_data['timestamp']
        ear_times = np.array([e['timestamp'] - base_time for e in ear_ts])
        ear_values = np.array([e['ear'] for e in ear_ts])
        ear_states = [e['state'] for e in ear_ts]
        
        ellipse_times = np.array([e['timestamp'] - base_time for e in ellipse_ts])
        major_axes = np.array([e['major_axis'] for e in ellipse_ts])
        minor_axes = np.array([e['minor_axis'] for e in ellipse_ts])
        areas = np.array([e['area'] for e in ellipse_ts])
        angles = np.array([e['angle'] for e in ellipse_ts])
        eccentricities = np.array([e['eccentricity'] for e in ellipse_ts])
        ellipse_states = [e['state'] for e in ellipse_ts]
        
        # 1. EAR時系列
        ax = axes[0, 0]
        self._plot_with_states(ax, ear_times, ear_values, ear_states, 
                              'EAR Time Series', 'Time (s)', 'EAR')
        ax.axhline(y=0.21, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()
        
        # 2. 楕円長軸
        ax = axes[0, 1]
        self._plot_with_states(ax, ellipse_times, major_axes, ellipse_states,
                              'Major Axis Time Series', 'Time (s)', 'Major Axis (px)')
        
        # 3. 楕円短軸
        ax = axes[1, 0]
        self._plot_with_states(ax, ellipse_times, minor_axes, ellipse_states,
                              'Minor Axis Time Series', 'Time (s)', 'Minor Axis (px)')
        
        # 4. 楕円面積
        ax = axes[1, 1]
        self._plot_with_states(ax, ellipse_times, areas, ellipse_states,
                              'Area Time Series', 'Time (s)', 'Area (px²)')
        
        # 5. 楕円角度
        ax = axes[2, 0]
        self._plot_with_states(ax, ellipse_times, angles, ellipse_states,
                              'Angle Time Series', 'Time (s)', 'Angle (°)')
        
        # 6. 楕円偏心率
        ax = axes[2, 1]
        self._plot_with_states(ax, ellipse_times, eccentricities, ellipse_states,
                              'Eccentricity Time Series', 'Time (s)', 'Eccentricity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"グラフを保存しました: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_with_states(self, ax, times, values, states, title, xlabel, ylabel):
        """
        状態を色分けしてプロット
        """
        # データポイントのプロット
        ax.plot(times, values, 'o-', linewidth=2, markersize=6, color='navy', alpha=0.7)
        
        # 背景に状態を色分け
        for i in range(len(times) - 1):
            state = states[i]
            color = self.state_colors.get(state, 'white')
            rect = Rectangle((times[i], ax.get_ylim()[0]), 
                           times[i+1] - times[i], 
                           ax.get_ylim()[1] - ax.get_ylim()[0],
                           facecolor=color, alpha=0.3, zorder=0)
            ax.add_patch(rect)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    
    def visualize_multiple_blinks(self, session_data, max_blinks=5, save_path=None):
        """
        複数の瞬きのEARと楕円パラメータを比較
        
        Args:
            session_data (dict): セッションデータ
            max_blinks (int): 表示する瞬きの最大数
            save_path (str): 保存先パス
        """
        blinks = session_data['blinks'][:max_blinks]
        
        if len(blinks) == 0:
            print("瞬きデータがありません")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Session: {session_data["session_id"]} - Multiple Blinks Comparison', 
                     fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(blinks)))
        
        for i, blink in enumerate(blinks):
            ear_ts = blink['ear_timeseries']
            ellipse_ts = blink['ellipse_timeseries']
            
            if len(ear_ts) == 0 or len(ellipse_ts) == 0:
                continue
            
            base_time = blink['timestamp']
            
            # EAR
            ear_times = [e['timestamp'] - base_time for e in ear_ts]
            ear_values = [e['ear'] for e in ear_ts]
            axes[0, 0].plot(ear_times, ear_values, 'o-', color=colors[i], 
                          label=f'Blink {blink["blink_id"]}', alpha=0.7)
            
            # 楕円長軸
            ellipse_times = [e['timestamp'] - base_time for e in ellipse_ts]
            major_axes = [e['major_axis'] for e in ellipse_ts]
            axes[0, 1].plot(ellipse_times, major_axes, 'o-', color=colors[i], 
                          label=f'Blink {blink["blink_id"]}', alpha=0.7)
            
            # 楕円短軸
            minor_axes = [e['minor_axis'] for e in ellipse_ts]
            axes[1, 0].plot(ellipse_times, minor_axes, 'o-', color=colors[i], 
                          label=f'Blink {blink["blink_id"]}', alpha=0.7)
            
            # 楕円面積
            areas = [e['area'] for e in ellipse_ts]
            axes[1, 1].plot(ellipse_times, areas, 'o-', color=colors[i], 
                          label=f'Blink {blink["blink_id"]}', alpha=0.7)
        
        # グラフの設定
        axes[0, 0].set_title('EAR Comparison', fontweight='bold')
        axes[0, 0].set_xlabel('Relative Time (s)')
        axes[0, 0].set_ylabel('EAR')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Major Axis Comparison', fontweight='bold')
        axes[0, 1].set_xlabel('Relative Time (s)')
        axes[0, 1].set_ylabel('Major Axis (px)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Minor Axis Comparison', fontweight='bold')
        axes[1, 0].set_xlabel('Relative Time (s)')
        axes[1, 0].set_ylabel('Minor Axis (px)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Area Comparison', fontweight='bold')
        axes[1, 1].set_xlabel('Relative Time (s)')
        axes[1, 1].set_ylabel('Area (px²)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"比較グラフを保存しました: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_normal_vs_drowsy(self, normal_data, drowsy_data, save_path=None):
        """
        正常状態と眠気状態の瞬きパターンを比較
        
        Args:
            normal_data (dict): 正常状態のセッションデータ
            drowsy_data (dict): 眠気状態のセッションデータ
            save_path (str): 保存先パス
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Normal vs Drowsy - Blink Pattern Comparison', 
                     fontsize=16, fontweight='bold')
        
        # 統計量の比較
        normal_stats = self._extract_statistics(normal_data)
        drowsy_stats = self._extract_statistics(drowsy_data)
        
        metrics = [
            ('Blink Coefficient', 'blink_coefficient'),
            ('Closing Time (ms)', 'closing_time', 1000),
            ('Opening Time (ms)', 'opening_time', 1000),
            ('EAR Min', 'ear_min'),
            ('Major Axis Max (px)', 'ellipse_major_axis_max'),
            ('Minor Axis Min (px)', 'ellipse_minor_axis_min')
        ]
        
        for i, metric_info in enumerate(metrics):
            ax = axes[i // 3, i % 3]
            
            if len(metric_info) == 3:
                metric_name, key, scale = metric_info
            else:
                metric_name, key = metric_info
                scale = 1
            
            normal_values = [s[key] * scale for s in normal_stats]
            drowsy_values = [s[key] * scale for s in drowsy_stats]
            
            ax.boxplot([normal_values, drowsy_values], 
                      labels=['Normal', 'Drowsy'])
            ax.set_title(metric_name, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"比較グラフを保存しました: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _extract_statistics(self, session_data):
        """セッションデータから統計量を抽出"""
        return [blink['statistics'] for blink in session_data['blinks'] 
                if blink['statistics']]


# ===== 使用例 =====
if __name__ == "__main__":
    import os
    
    # 可視化ツールの初期化
    visualizer = BlinkTimeSeriesVisualizer()
    
    # サンプルデータの読み込み
    data_dir = "data/blink_sessions_timeseries"
    
    if os.path.exists(data_dir):
        # 最初のJSONファイルを読み込み
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        
        if len(json_files) > 0:
            filepath = os.path.join(data_dir, json_files[0])
            
            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # 単一瞬きの可視化
            if len(session_data['blinks']) > 0:
                visualizer.visualize_single_blink(
                    session_data['blinks'][0],
                    save_path=f"{data_dir}/blink_1_analysis.png"
                )
            
            # 複数瞬きの比較
            visualizer.visualize_multiple_blinks(
                session_data,
                max_blinks=5,
                save_path=f"{data_dir}/multiple_blinks_comparison.png"
            )
        else:
            print("JSONファイルが見つかりません")
    else:
        print(f"ディレクトリが存在しません: {data_dir}")
