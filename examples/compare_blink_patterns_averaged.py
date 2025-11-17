"""
眠気検出システム - 実時間スケール瞬き比較プログラム（最終改訂版）

EAR最小値を時間0として、実際の時間（ミリ秒）で開眼・閉眼過程を表示。
- 閉眼過程: 負の時間（-50ms → 0ms）
- 開眼過程: 正の時間（0ms → +600ms）
- 実際の時間差が視覚的に明確

使用方法:
    python compare_blink_real_time.py normal.json drowsy.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import interpolate
import argparse


class BlinkRealTimeComparator:
    """実時間スケールでの瞬き比較"""
    
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
    
    def extract_blink_with_real_time(self, blink):
        """
        瞬きデータを実時間付きで抽出
        EAR最小値を時間0として、閉眼過程は負、開眼過程は正の時間で返す
        
        Returns:
            dict: {
                'closing_times': [...],  # 負の時間（ms）
                'closing_ear': [...],
                'closing_vd': [...],
                'opening_times': [...],  # 正の時間（ms）
                'opening_ear': [...],
                'opening_vd': [...],
                'closing_duration': float,  # ms
                'opening_duration': float   # ms
            }
        """
        ear_ts = blink.get('ear_timeseries', [])
        circles_ts = blink.get('circles_timeseries', [])
        
        if len(ear_ts) < 2:
            return None
        
        # EAR最小値のインデックス
        ear_values = [p['ear'] for p in ear_ts]
        min_idx = np.argmin(ear_values)
        
        # タイムスタンプを取得（秒）
        timestamps = [p['timestamp'] for p in ear_ts]
        min_timestamp = timestamps[min_idx]
        
        # EAR最小値を0として相対時間に変換（ミリ秒）
        relative_times = [(t - min_timestamp) * 1000 for t in timestamps]
        
        # 閉眼過程と開眼過程に分割
        closing_times = relative_times[:min_idx + 1]
        closing_ear = ear_values[:min_idx + 1]
        
        opening_times = relative_times[min_idx:]
        opening_ear = ear_values[min_idx:]
        
        # 垂直距離も同様に抽出
        closing_vd = []
        opening_vd = []
        
        if len(circles_ts) >= len(ear_ts):
            vd_values = [p['vertical_distance'] for p in circles_ts]
            min_idx_circles = min(min_idx, len(vd_values) - 1)
            closing_vd = vd_values[:min_idx_circles + 1]
            opening_vd = vd_values[min_idx_circles:]
        
        # 継続時間を計算
        closing_duration = abs(closing_times[0]) if len(closing_times) > 0 else 0
        opening_duration = opening_times[-1] if len(opening_times) > 0 else 0
        
        return {
            'closing_times': closing_times,
            'closing_ear': closing_ear,
            'closing_vd': closing_vd,
            'opening_times': opening_times,
            'opening_ear': opening_ear,
            'opening_vd': opening_vd,
            'closing_duration': closing_duration,
            'opening_duration': opening_duration
        }
    
    def align_and_average_real_time(self, blink_data_list, phase='opening', 
                                    param='ear', time_range=None):
        """
        実時間データを揃えて平均を計算
        
        Args:
            blink_data_list: 瞬きデータのリスト
            phase: 'opening' or 'closing'
            param: 'ear' or 'vd'
            time_range: (min_time, max_time) in ms
        
        Returns:
            times: 時間軸（ms）
            mean_values: 平均値
            std_values: 標準偏差
        """
        if time_range is None:
            # 自動的に範囲を決定
            all_times = []
            for data in blink_data_list:
                if data is None:
                    continue
                times = data[f'{phase}_times']
                all_times.extend(times)
            
            if len(all_times) == 0:
                return np.array([]), np.array([]), np.array([])
            
            time_range = (min(all_times), max(all_times))
        
        # 共通の時間軸を作成
        common_times = np.linspace(time_range[0], time_range[1], 200)
        interpolated_values = []
        
        for data in blink_data_list:
            if data is None:
                continue
            
            times = data[f'{phase}_times']
            values = data[f'{phase}_{param}']
            
            if len(times) < 2 or len(values) < 2:
                continue
            
            # 線形補間
            f = interpolate.interp1d(times, values, kind='linear', 
                                    bounds_error=False, fill_value=np.nan)
            interpolated = f(common_times)
            
            # NaNを除外
            if not np.all(np.isnan(interpolated)):
                interpolated_values.append(interpolated)
        
        if len(interpolated_values) == 0:
            return common_times, np.zeros_like(common_times), np.zeros_like(common_times)
        
        # 平均と標準偏差（NaNを無視）
        interpolated_array = np.array(interpolated_values)
        mean_values = np.nanmean(interpolated_array, axis=0)
        std_values = np.nanstd(interpolated_array, axis=0)
        
        return common_times, mean_values, std_values
    
    def create_real_time_comparison(self, save_path=None):
        """実時間スケールでの比較グラフを作成"""
        
        print("\n" + "="*80)
        print("実時間スケールでの瞬き分析中...")
        print("="*80)
        
        normal_blinks = self.normal_data['blinks']
        drowsy_blinks = self.drowsy_data['blinks']
        
        # 全瞬きから実時間データを抽出
        normal_data = []
        drowsy_data = []
        
        for blink in normal_blinks:
            data = self.extract_blink_with_real_time(blink)
            if data is not None:
                normal_data.append(data)
        
        for blink in drowsy_blinks:
            data = self.extract_blink_with_real_time(blink)
            if data is not None:
                drowsy_data.append(data)
        
        print(f"\n通常時: {len(normal_data)}個の瞬きを抽出")
        print(f"眠気時: {len(drowsy_data)}個の瞬きを抽出")
        
        # 統計情報を計算
        normal_closing_durations = [d['closing_duration'] for d in normal_data]
        normal_opening_durations = [d['opening_duration'] for d in normal_data]
        drowsy_closing_durations = [d['closing_duration'] for d in drowsy_data]
        drowsy_opening_durations = [d['opening_duration'] for d in drowsy_data]
        
        print(f"\n【閉眼時間】")
        print(f"  通常時: {np.mean(normal_closing_durations):.1f} ± {np.std(normal_closing_durations):.1f} ms")
        print(f"  眠気時: {np.mean(drowsy_closing_durations):.1f} ± {np.std(drowsy_closing_durations):.1f} ms")
        
        print(f"\n【開眼時間】")
        print(f"  通常時: {np.mean(normal_opening_durations):.1f} ± {np.std(normal_opening_durations):.1f} ms")
        print(f"  眠気時: {np.mean(drowsy_opening_durations):.1f} ± {np.std(drowsy_opening_durations):.1f} ms")
        
        # 時間範囲を決定
        max_closing = max(max(normal_closing_durations), max(drowsy_closing_durations))
        max_opening = max(max(normal_opening_durations), max(drowsy_opening_durations))
        
        closing_range = (-max_closing * 1.1, 0)
        opening_range = (0, max_opening * 1.1)
        
        # 平均曲線を計算
        # 閉眼過程 - EAR
        normal_closing_times, normal_closing_ear_mean, normal_closing_ear_std = \
            self.align_and_average_real_time(normal_data, 'closing', 'ear', closing_range)
        drowsy_closing_times, drowsy_closing_ear_mean, drowsy_closing_ear_std = \
            self.align_and_average_real_time(drowsy_data, 'closing', 'ear', closing_range)
        
        # 開眼過程 - EAR
        normal_opening_times, normal_opening_ear_mean, normal_opening_ear_std = \
            self.align_and_average_real_time(normal_data, 'opening', 'ear', opening_range)
        drowsy_opening_times, drowsy_opening_ear_mean, drowsy_opening_ear_std = \
            self.align_and_average_real_time(drowsy_data, 'opening', 'ear', opening_range)
        
        # 閉眼過程 - 垂直距離
        normal_closing_vd_times, normal_closing_vd_mean, normal_closing_vd_std = \
            self.align_and_average_real_time(normal_data, 'closing', 'vd', closing_range)
        drowsy_closing_vd_times, drowsy_closing_vd_mean, drowsy_closing_vd_std = \
            self.align_and_average_real_time(drowsy_data, 'closing', 'vd', closing_range)
        
        # 開眼過程 - 垂直距離
        normal_opening_vd_times, normal_opening_vd_mean, normal_opening_vd_std = \
            self.align_and_average_real_time(normal_data, 'opening', 'vd', opening_range)
        drowsy_opening_vd_times, drowsy_opening_vd_mean, drowsy_opening_vd_std = \
            self.align_and_average_real_time(drowsy_data, 'opening', 'vd', opening_range)
        
        print("✓ 平均曲線の計算完了\n")
        
        # グラフ作成
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        fig.suptitle(f'Real-Time Scale Comparison: Normal (n={len(normal_data)}) vs Drowsy (n={len(drowsy_data)})\n'
                     f'Time 0 = EAR Minimum Point',
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. 閉眼過程 - EAR
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(normal_closing_times, normal_closing_ear_mean, color='blue', linewidth=2.5,
                label=f'Normal (KSS={self.normal_data["kss_score"]})')
        ax1.fill_between(normal_closing_times,
                         normal_closing_ear_mean - normal_closing_ear_std,
                         normal_closing_ear_mean + normal_closing_ear_std,
                         color='blue', alpha=0.2, label='± SD')
        
        ax1.plot(drowsy_closing_times, drowsy_closing_ear_mean, color='red', linewidth=2.5,
                label=f'Drowsy (KSS={self.drowsy_data["kss_score"]})')
        ax1.fill_between(drowsy_closing_times,
                         drowsy_closing_ear_mean - drowsy_closing_ear_std,
                         drowsy_closing_ear_mean + drowsy_closing_ear_std,
                         color='red', alpha=0.2)
        
        ax1.axhline(y=0.21, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
        ax1.axvline(x=0, color='gray', linestyle='-', linewidth=1.5, alpha=0.7, label='EAR Min (t=0)')
        ax1.set_title('Closing Phase: EAR (Real Time)', fontweight='bold', fontsize=13)
        ax1.set_xlabel('Time relative to EAR Minimum (ms)\n← Closing starts', fontweight='bold')
        ax1.set_ylabel('EAR', fontweight='bold')
        ax1.legend(fontsize=9, loc='upper left')
        ax1.grid(alpha=0.3, linestyle='--')
        
        # 2. 開眼過程 - EAR ★最重要★
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(normal_opening_times, normal_opening_ear_mean, color='blue', linewidth=2.5,
                label=f'Normal (KSS={self.normal_data["kss_score"]})')
        ax2.fill_between(normal_opening_times,
                         normal_opening_ear_mean - normal_opening_ear_std,
                         normal_opening_ear_mean + normal_opening_ear_std,
                         color='blue', alpha=0.2, label='± SD')
        
        ax2.plot(drowsy_opening_times, drowsy_opening_ear_mean, color='red', linewidth=2.5,
                label=f'Drowsy (KSS={self.drowsy_data["kss_score"]})')
        ax2.fill_between(drowsy_opening_times,
                         drowsy_opening_ear_mean - drowsy_opening_ear_std,
                         drowsy_opening_ear_mean + drowsy_opening_ear_std,
                         color='red', alpha=0.2)
        
        ax2.axhline(y=0.21, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
        ax2.axvline(x=0, color='gray', linestyle='-', linewidth=1.5, alpha=0.7, label='EAR Min (t=0)')
        ax2.set_title('Opening Phase: EAR (Real Time) ★TIME DIFFERENCE VISIBLE★',
                     fontweight='bold', fontsize=13, color='darkred')
        ax2.set_xlabel('Time relative to EAR Minimum (ms)\nOpening ends →', fontweight='bold')
        ax2.set_ylabel('EAR', fontweight='bold')
        ax2.legend(fontsize=9, loc='lower right')
        ax2.grid(alpha=0.3, linestyle='--')
        
        # 3. 閉眼過程 - 垂直距離
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(normal_closing_vd_times, normal_closing_vd_mean, color='blue', linewidth=2.5,
                label=f'Normal')
        ax3.fill_between(normal_closing_vd_times,
                         normal_closing_vd_mean - normal_closing_vd_std,
                         normal_closing_vd_mean + normal_closing_vd_std,
                         color='blue', alpha=0.2, label='± SD')
        
        ax3.plot(drowsy_closing_vd_times, drowsy_closing_vd_mean, color='red', linewidth=2.5,
                label=f'Drowsy')
        ax3.fill_between(drowsy_closing_vd_times,
                         drowsy_closing_vd_mean - drowsy_closing_vd_std,
                         drowsy_closing_vd_mean + drowsy_closing_vd_std,
                         color='red', alpha=0.2)
        
        ax3.axvline(x=0, color='gray', linestyle='-', linewidth=1.5, alpha=0.7, label='Min (t=0)')
        ax3.set_title('Closing Phase: Vertical Distance (Real Time)', fontweight='bold', fontsize=13)
        ax3.set_xlabel('Time relative to Minimum (ms)\n← Closing starts', fontweight='bold')
        ax3.set_ylabel('Distance (px)', fontweight='bold')
        ax3.legend(fontsize=9, loc='upper left')
        ax3.grid(alpha=0.3, linestyle='--')
        
        # 4. 開眼過程 - 垂直距離 ★重要★
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(normal_opening_vd_times, normal_opening_vd_mean, color='blue', linewidth=2.5,
                label=f'Normal')
        ax4.fill_between(normal_opening_vd_times,
                         normal_opening_vd_mean - normal_opening_vd_std,
                         normal_opening_vd_mean + normal_opening_vd_std,
                         color='blue', alpha=0.2, label='± SD')
        
        ax4.plot(drowsy_opening_vd_times, drowsy_opening_vd_mean, color='red', linewidth=2.5,
                label=f'Drowsy')
        ax4.fill_between(drowsy_opening_vd_times,
                         drowsy_opening_vd_mean - drowsy_opening_vd_std,
                         drowsy_opening_vd_mean + drowsy_opening_vd_std,
                         color='red', alpha=0.2)
        
        ax4.axvline(x=0, color='gray', linestyle='-', linewidth=1.5, alpha=0.7, label='Min (t=0)')
        ax4.set_title('Opening Phase: Vertical Distance (Real Time) ★TIME DIFFERENCE VISIBLE★',
                     fontweight='bold', fontsize=13, color='darkred')
        ax4.set_xlabel('Time relative to Minimum (ms)\nOpening ends →', fontweight='bold')
        ax4.set_ylabel('Distance (px)', fontweight='bold')
        ax4.legend(fontsize=9, loc='lower right')
        ax4.grid(alpha=0.3, linestyle='--')
        
        # 統計情報を追加
        stats_text = (
            f'Time Statistics (Mean ± SD):\n'
            f'Closing Time:  Normal {np.mean(normal_closing_durations):.0f}±{np.std(normal_closing_durations):.0f}ms  '
            f'Drowsy {np.mean(drowsy_closing_durations):.0f}±{np.std(drowsy_closing_durations):.0f}ms\n'
            f'Opening Time:  Normal {np.mean(normal_opening_durations):.0f}±{np.std(normal_opening_durations):.0f}ms  '
            f'Drowsy {np.mean(drowsy_opening_durations):.0f}±{np.std(drowsy_opening_durations):.0f}ms  '
            f'(Drowsy is {np.mean(drowsy_opening_durations)/np.mean(normal_opening_durations):.1f}x longer!)'
        )
        
        fig.text(0.5, 0.01, stats_text,
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ グラフを保存しました: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='実時間スケールでの瞬き比較')
    parser.add_argument('normal_json', help='通常時のJSONファイルパス')
    parser.add_argument('drowsy_json', help='眠気時のJSONファイルパス')
    parser.add_argument('--output-dir', default='graph/average/outputs',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 比較オブジェクトの作成
    comparator = BlinkRealTimeComparator(args.normal_json, args.drowsy_json)
    
    # 実時間比較グラフ
    print("\n" + "="*80)
    print("実時間スケール比較グラフを生成中...")
    print("="*80)
    save_path = output_dir / 'blink_real_time_comparison.png'
    comparator.create_real_time_comparison(save_path)
    
    print("\n" + "="*80)
    print("✓ グラフ生成が完了しました！")
    print("="*80)
    print(f"\n生成されたファイル:")
    print(f"  {save_path}")
    print()


if __name__ == '__main__':
    main()