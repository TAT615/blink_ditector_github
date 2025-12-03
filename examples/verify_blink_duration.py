#!/usr/bin/env python3
"""
瞬き時間検証プログラム

収集したデータの瞬きが実際に伸びているかを検証します。
- 閉眼時間（Closing Time）：閉じ始め→完全閉眼
- 開眼時間（Opening Time）：完全閉眼→瞬き終了

UserIDを入力し、そのユーザーのnormalとdrowsyデータを比較します。

使用方法:
    python verify_blink_duration.py

出力:
    - 比較グラフ（箱ひげ図、ヒストグラム、散布図）
    - 統計サマリー
"""

import json
import os
import glob
import numpy as np
import sys

# ヘッドレス環境チェック
if 'DISPLAY' not in os.environ and sys.platform != 'win32' and sys.platform != 'darwin':
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
import platform
if platform.system() == 'Windows':
    rcParams['font.family'] = 'MS Gothic'
elif platform.system() == 'Darwin':  # macOS
    rcParams['font.family'] = 'Hiragino Sans'
else:
    rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class BlinkDurationVerifier:
    """瞬き時間検証クラス"""
    
    def __init__(self, data_dir="data"):
        """
        初期化
        
        Args:
            data_dir: データディレクトリのパス
        """
        self.data_dir = data_dir
        self.sessions_dir = os.path.join(data_dir, "sessions")
        
        # データ格納用
        self.normal_data = {
            'closing_time': [],
            'opening_time': [],
            'total_duration': [],
            'blink_coefficient': []
        }
        self.drowsy_data = {
            'closing_time': [],
            'opening_time': [],
            'total_duration': [],
            'blink_coefficient': []
        }
        
        self.user_id = None
        
    def list_available_users(self):
        """利用可能なユーザーIDの一覧を取得"""
        if not os.path.exists(self.sessions_dir):
            print(f"❌ データディレクトリが見つかりません: {self.sessions_dir}")
            return []
        
        # JSONファイルからユーザーIDを抽出
        json_files = glob.glob(os.path.join(self.sessions_dir, "*.json"))
        user_ids = set()
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'user_id' in data:
                        user_ids.add(data['user_id'])
            except (json.JSONDecodeError, KeyError):
                continue
        
        return sorted(list(user_ids))
    
    def load_user_data(self, user_id):
        """
        指定したユーザーIDのデータを読み込み
        
        Args:
            user_id: ユーザーID
            
        Returns:
            bool: 読み込み成功かどうか
        """
        self.user_id = user_id
        
        # データをリセット
        self.normal_data = {
            'closing_time': [],
            'opening_time': [],
            'total_duration': [],
            'blink_coefficient': []
        }
        self.drowsy_data = {
            'closing_time': [],
            'opening_time': [],
            'total_duration': [],
            'blink_coefficient': []
        }
        
        # JSONファイルを検索
        json_files = glob.glob(os.path.join(self.sessions_dir, "*.json"))
        
        normal_sessions = 0
        drowsy_sessions = 0
        normal_blinks = 0
        drowsy_blinks = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # ユーザーIDが一致するか確認
                if data.get('user_id') != user_id:
                    continue
                
                # ラベルを確認
                label = data.get('label', -1)
                blinks = data.get('blinks', [])
                
                if label == 0:  # Normal
                    target_data = self.normal_data
                    normal_sessions += 1
                elif label == 1:  # Drowsy
                    target_data = self.drowsy_data
                    drowsy_sessions += 1
                else:
                    continue
                
                # 各瞬きのデータを抽出
                for blink in blinks:
                    # 時間データの抽出（異なる形式に対応）
                    closing_time = None
                    opening_time = None
                    total_duration = None
                    blink_coefficient = None
                    
                    # 形式1: statistics内にある場合
                    if 'statistics' in blink:
                        stats = blink['statistics']
                        closing_time = stats.get('closing_time')
                        opening_time = stats.get('opening_time')
                        total_duration = stats.get('total_duration')
                        blink_coefficient = stats.get('blink_coefficient')
                    
                    # 形式2: 直接ある場合
                    if closing_time is None:
                        closing_time = blink.get('closing_time')
                    if opening_time is None:
                        opening_time = blink.get('opening_time')
                    if total_duration is None:
                        total_duration = blink.get('total_duration')
                    if blink_coefficient is None:
                        blink_coefficient = blink.get('blink_coefficient')
                    
                    # 有効なデータのみ追加
                    if closing_time is not None and opening_time is not None:
                        # ミリ秒に変換（秒単位の場合）
                        if closing_time < 1:  # 秒単位と判断
                            closing_time *= 1000
                            opening_time *= 1000
                            if total_duration is not None:
                                total_duration *= 1000
                        
                        target_data['closing_time'].append(closing_time)
                        target_data['opening_time'].append(opening_time)
                        
                        if total_duration is not None:
                            target_data['total_duration'].append(total_duration)
                        if blink_coefficient is not None:
                            target_data['blink_coefficient'].append(blink_coefficient)
                        
                        if label == 0:
                            normal_blinks += 1
                        else:
                            drowsy_blinks += 1
                            
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️ ファイル読み込みエラー: {json_file} - {e}")
                continue
        
        print(f"\n📊 ユーザー '{user_id}' のデータ読み込み完了:")
        print(f"   正常状態: {normal_sessions} セッション, {normal_blinks} 瞬き")
        print(f"   眠気状態: {drowsy_sessions} セッション, {drowsy_blinks} 瞬き")
        
        return normal_blinks > 0 or drowsy_blinks > 0
    
    def calculate_statistics(self):
        """統計量を計算"""
        stats = {}
        
        for label, data in [('normal', self.normal_data), ('drowsy', self.drowsy_data)]:
            stats[label] = {}
            
            for key in ['closing_time', 'opening_time', 'total_duration', 'blink_coefficient']:
                values = data[key]
                if len(values) > 0:
                    stats[label][key] = {
                        'n': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75)
                    }
        
        return stats
    
    def print_statistics(self):
        """統計サマリーを表示"""
        stats = self.calculate_statistics()
        
        print("\n" + "=" * 70)
        print(f"📈 瞬き時間の統計サマリー - ユーザー: {self.user_id}")
        print("=" * 70)
        
        # 比較表を作成
        metrics = [
            ('closing_time', 'Closing Time (Tc)', 'ms'),
            ('opening_time', 'Opening Time (To)', 'ms'),
            ('total_duration', 'Total Duration', 'ms'),
            ('blink_coefficient', 'Blink Coefficient (To/Tc)', '')
        ]
        
        for key, name, unit in metrics:
            print(f"\n【{name}】")
            print("-" * 50)
            
            for label in ['normal', 'drowsy']:
                label_name = 'Normal (正常)' if label == 'normal' else 'Drowsy (眠気)'
                
                if label in stats and key in stats[label]:
                    s = stats[label][key]
                    unit_str = f" {unit}" if unit else ""
                    print(f"  {label_name}:")
                    print(f"    サンプル数: {s['n']}")
                    print(f"    平均 ± SD:  {s['mean']:.2f} ± {s['std']:.2f}{unit_str}")
                    print(f"    中央値:     {s['median']:.2f}{unit_str}")
                    print(f"    範囲:       {s['min']:.2f} - {s['max']:.2f}{unit_str}")
                else:
                    print(f"  {label_name}: データなし")
            
            # 変化率を計算
            if ('normal' in stats and key in stats['normal'] and 
                'drowsy' in stats and key in stats['drowsy']):
                normal_mean = stats['normal'][key]['mean']
                drowsy_mean = stats['drowsy'][key]['mean']
                change = ((drowsy_mean - normal_mean) / normal_mean) * 100
                print(f"\n  変化率: {change:+.1f}% (正常 → 眠気)")
        
        print("\n" + "=" * 70)
    
    def perform_statistical_tests(self):
        """
        統計的検定を実施
        - Welch's t検定（等分散を仮定しない）
        - Mann-Whitney U検定（ノンパラメトリック）
        - Cohen's d（効果量）
        """
        print("\n" + "=" * 70)
        print(f"📊 統計検定結果 - ユーザー: {self.user_id}")
        print("=" * 70)
        
        has_normal = len(self.normal_data['closing_time']) > 0
        has_drowsy = len(self.drowsy_data['closing_time']) > 0
        
        if not (has_normal and has_drowsy):
            print("⚠️ 両群のデータが必要です（NormalとDrowsy）")
            return
        
        metrics = [
            ('closing_time', 'Closing Time (Tc)'),
            ('opening_time', 'Opening Time (To)'),
            ('total_duration', 'Total Duration'),
            ('blink_coefficient', 'Blink Coefficient (To/Tc)')
        ]
        
        print("\n有意水準: α = 0.05")
        print("-" * 70)
        
        results = {}
        
        for key, name in metrics:
            normal_vals = np.array(self.normal_data[key])
            drowsy_vals = np.array(self.drowsy_data[key])
            
            if len(normal_vals) < 2 or len(drowsy_vals) < 2:
                continue
            
            print(f"\n【{name}】")
            
            # Welch's t検定
            t_stat, t_pvalue = scipy_stats.ttest_ind(normal_vals, drowsy_vals, equal_var=False)
            
            # Mann-Whitney U検定（ノンパラメトリック）
            u_stat, u_pvalue = scipy_stats.mannwhitneyu(normal_vals, drowsy_vals, alternative='two-sided')
            
            # 効果量（Cohen's d）
            pooled_std = np.sqrt(((len(normal_vals)-1)*np.std(normal_vals, ddof=1)**2 + 
                                  (len(drowsy_vals)-1)*np.std(drowsy_vals, ddof=1)**2) / 
                                 (len(normal_vals) + len(drowsy_vals) - 2))
            cohens_d = (np.mean(drowsy_vals) - np.mean(normal_vals)) / pooled_std if pooled_std > 0 else 0
            
            # 効果量の解釈
            if abs(cohens_d) < 0.2:
                effect_interp = "negligible (無視できる)"
            elif abs(cohens_d) < 0.5:
                effect_interp = "small (小)"
            elif abs(cohens_d) < 0.8:
                effect_interp = "medium (中)"
            else:
                effect_interp = "large (大)"
            
            print(f"  サンプルサイズ: Normal={len(normal_vals)}, Drowsy={len(drowsy_vals)}")
            print(f"  平均値: Normal={np.mean(normal_vals):.2f}, Drowsy={np.mean(drowsy_vals):.2f}")
            print()
            print(f"  Welch's t-test:")
            print(f"    t統計量 = {t_stat:.4f}")
            print(f"    p値     = {t_pvalue:.6f} {'***' if t_pvalue < 0.001 else '**' if t_pvalue < 0.01 else '*' if t_pvalue < 0.05 else ''}")
            print()
            print(f"  Mann-Whitney U test:")
            print(f"    U統計量 = {u_stat:.4f}")
            print(f"    p値     = {u_pvalue:.6f} {'***' if u_pvalue < 0.001 else '**' if u_pvalue < 0.01 else '*' if u_pvalue < 0.05 else ''}")
            print()
            print(f"  効果量 (Cohen's d) = {cohens_d:.4f} [{effect_interp}]")
            
            # 結論
            if t_pvalue < 0.05:
                direction = "増加" if np.mean(drowsy_vals) > np.mean(normal_vals) else "減少"
                print(f"\n  → 結論: 眠気状態で有意に{direction}しています (p < 0.05)")
            else:
                print(f"\n  → 結論: 有意差は認められません (p >= 0.05)")
            
            results[key] = {
                't_stat': t_stat,
                't_pvalue': t_pvalue,
                'u_stat': u_stat,
                'u_pvalue': u_pvalue,
                'cohens_d': cohens_d
            }
        
        print("\n" + "=" * 70)
        print("  * p < 0.05, ** p < 0.01, *** p < 0.001")
        print("=" * 70)
        
        return results
    
    def create_comparison_plot(self, output_dir="output"):
        """
        比較グラフを作成
        
        Args:
            output_dir: 出力ディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # データチェック
        has_normal = len(self.normal_data['closing_time']) > 0
        has_drowsy = len(self.drowsy_data['closing_time']) > 0
        
        if not has_normal and not has_drowsy:
            print("❌ プロット用のデータがありません")
            return
        
        # 大きな図を作成（2行 x 2列）
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Blink Duration Analysis - User: {self.user_id}', 
                     fontsize=16, fontweight='bold')
        
        # カラー設定
        normal_color = '#3498db'  # 青
        drowsy_color = '#e74c3c'  # 赤
        
        # ===== 1. 箱ひげ図（Closing Time vs Opening Time）=====
        ax1 = axes[0, 0]
        
        plot_data = []
        labels = []
        colors = []
        
        if has_normal:
            plot_data.extend([self.normal_data['closing_time'], 
                            self.normal_data['opening_time']])
            labels.extend(['Normal\nClosing', 'Normal\nOpening'])
            colors.extend([normal_color, normal_color])
        
        if has_drowsy:
            plot_data.extend([self.drowsy_data['closing_time'], 
                            self.drowsy_data['opening_time']])
            labels.extend(['Drowsy\nClosing', 'Drowsy\nOpening'])
            colors.extend([drowsy_color, drowsy_color])
        
        bp = ax1.boxplot(plot_data, labels=labels, patch_artist=True)
        
        for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('Closing vs Opening Time Distribution', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 平均値を点で表示
        for i, data in enumerate(plot_data):
            mean_val = np.mean(data)
            ax1.scatter(i + 1, mean_val, color='black', marker='D', s=50, zorder=5)
        
        # ===== 2. ヒストグラム（Closing Time）=====
        ax2 = axes[0, 1]
        
        bins = np.linspace(0, max(
            max(self.normal_data['closing_time']) if has_normal else 0,
            max(self.drowsy_data['closing_time']) if has_drowsy else 0
        ) * 1.1, 30)
        
        if has_normal:
            ax2.hist(self.normal_data['closing_time'], bins=bins, alpha=0.6, 
                    color=normal_color, label=f'Normal (n={len(self.normal_data["closing_time"])})',
                    edgecolor='black', linewidth=0.5)
            ax2.axvline(np.mean(self.normal_data['closing_time']), color=normal_color, 
                       linestyle='--', linewidth=2, label=f'Normal Mean: {np.mean(self.normal_data["closing_time"]):.1f}ms')
        
        if has_drowsy:
            ax2.hist(self.drowsy_data['closing_time'], bins=bins, alpha=0.6, 
                    color=drowsy_color, label=f'Drowsy (n={len(self.drowsy_data["closing_time"])})',
                    edgecolor='black', linewidth=0.5)
            ax2.axvline(np.mean(self.drowsy_data['closing_time']), color=drowsy_color, 
                       linestyle='--', linewidth=2, label=f'Drowsy Mean: {np.mean(self.drowsy_data["closing_time"]):.1f}ms')
        
        ax2.set_xlabel('Closing Time (ms)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Closing Time Distribution (Tc)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # ===== 3. ヒストグラム（Opening Time）=====
        ax3 = axes[1, 0]
        
        bins = np.linspace(0, max(
            max(self.normal_data['opening_time']) if has_normal else 0,
            max(self.drowsy_data['opening_time']) if has_drowsy else 0
        ) * 1.1, 30)
        
        if has_normal:
            ax3.hist(self.normal_data['opening_time'], bins=bins, alpha=0.6, 
                    color=normal_color, label=f'Normal (n={len(self.normal_data["opening_time"])})',
                    edgecolor='black', linewidth=0.5)
            ax3.axvline(np.mean(self.normal_data['opening_time']), color=normal_color, 
                       linestyle='--', linewidth=2, label=f'Normal Mean: {np.mean(self.normal_data["opening_time"]):.1f}ms')
        
        if has_drowsy:
            ax3.hist(self.drowsy_data['opening_time'], bins=bins, alpha=0.6, 
                    color=drowsy_color, label=f'Drowsy (n={len(self.drowsy_data["opening_time"])})',
                    edgecolor='black', linewidth=0.5)
            ax3.axvline(np.mean(self.drowsy_data['opening_time']), color=drowsy_color, 
                       linestyle='--', linewidth=2, label=f'Drowsy Mean: {np.mean(self.drowsy_data["opening_time"]):.1f}ms')
        
        ax3.set_xlabel('Opening Time (ms)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Opening Time Distribution (To)', fontsize=12)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # ===== 4. 散布図（Closing vs Opening）=====
        ax4 = axes[1, 1]
        
        if has_normal:
            ax4.scatter(self.normal_data['closing_time'], self.normal_data['opening_time'],
                       alpha=0.5, color=normal_color, label='Normal', s=30)
        
        if has_drowsy:
            ax4.scatter(self.drowsy_data['closing_time'], self.drowsy_data['opening_time'],
                       alpha=0.5, color=drowsy_color, label='Drowsy', s=30)
        
        # 平均点をプロット
        if has_normal:
            ax4.scatter(np.mean(self.normal_data['closing_time']), 
                       np.mean(self.normal_data['opening_time']),
                       color=normal_color, marker='*', s=200, edgecolor='black', 
                       linewidth=1.5, label='Normal Mean', zorder=5)
        
        if has_drowsy:
            ax4.scatter(np.mean(self.drowsy_data['closing_time']), 
                       np.mean(self.drowsy_data['opening_time']),
                       color=drowsy_color, marker='*', s=200, edgecolor='black', 
                       linewidth=1.5, label='Drowsy Mean', zorder=5)
        
        ax4.set_xlabel('Closing Time (ms)', fontsize=12)
        ax4.set_ylabel('Opening Time (ms)', fontsize=12)
        ax4.set_title('Closing vs Opening Time Scatter', fontsize=12)
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # 対角線を追加（Tc = To）
        max_val = max(ax4.get_xlim()[1], ax4.get_ylim()[1])
        ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Tc = To')
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(output_dir, f"blink_duration_analysis_{self.user_id}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ グラフを保存しました: {output_path}")
        
        plt.show()
        
        return output_path
    
    def create_detailed_comparison_plot(self, output_dir="output"):
        """
        詳細な比較グラフを作成（平均バーグラフ + エラーバー）
        """
        os.makedirs(output_dir, exist_ok=True)
        
        stats = self.calculate_statistics()
        
        has_normal = 'normal' in stats and 'closing_time' in stats['normal']
        has_drowsy = 'drowsy' in stats and 'closing_time' in stats['drowsy']
        
        if not has_normal and not has_drowsy:
            print("❌ 詳細プロット用のデータがありません")
            return
        
        # 図を作成
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Mean Comparison with Standard Deviation - User: {self.user_id}', 
                     fontsize=14, fontweight='bold')
        
        # カラー設定
        normal_color = '#3498db'
        drowsy_color = '#e74c3c'
        
        # ===== 左: Closing Time と Opening Time の平均比較 =====
        ax1 = axes[0]
        
        x = np.arange(2)
        width = 0.35
        
        if has_normal and has_drowsy:
            # Normal
            normal_means = [stats['normal']['closing_time']['mean'], 
                          stats['normal']['opening_time']['mean']]
            normal_stds = [stats['normal']['closing_time']['std'], 
                         stats['normal']['opening_time']['std']]
            
            # Drowsy
            drowsy_means = [stats['drowsy']['closing_time']['mean'], 
                          stats['drowsy']['opening_time']['mean']]
            drowsy_stds = [stats['drowsy']['closing_time']['std'], 
                         stats['drowsy']['opening_time']['std']]
            
            bars1 = ax1.bar(x - width/2, normal_means, width, yerr=normal_stds, 
                           label='Normal', color=normal_color, alpha=0.7,
                           capsize=5, edgecolor='black')
            bars2 = ax1.bar(x + width/2, drowsy_means, width, yerr=drowsy_stds, 
                           label='Drowsy', color=drowsy_color, alpha=0.7,
                           capsize=5, edgecolor='black')
            
            # 値をバーの上に表示
            for bar, mean in zip(bars1, normal_means):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + normal_stds[0] + 5,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
            
            for bar, mean in zip(bars2, drowsy_means):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + drowsy_stds[0] + 5,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('Mean Closing & Opening Time', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Closing Time (Tc)', 'Opening Time (To)'])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # ===== 右: 変化率を表示 =====
        ax2 = axes[1]
        
        if has_normal and has_drowsy:
            metrics = ['closing_time', 'opening_time', 'total_duration']
            metric_names = ['Closing Time\n(Tc)', 'Opening Time\n(To)', 'Total Duration']
            changes = []
            
            for metric in metrics:
                if metric in stats['normal'] and metric in stats['drowsy']:
                    normal_mean = stats['normal'][metric]['mean']
                    drowsy_mean = stats['drowsy'][metric]['mean']
                    change = ((drowsy_mean - normal_mean) / normal_mean) * 100
                    changes.append(change)
                else:
                    changes.append(0)
            
            colors = [drowsy_color if c > 0 else normal_color for c in changes]
            bars = ax2.bar(metric_names, changes, color=colors, alpha=0.7, edgecolor='black')
            
            # 値を表示
            for bar, change in zip(bars, changes):
                y_pos = bar.get_height() + 2 if change > 0 else bar.get_height() - 5
                ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                        f'{change:+.1f}%', ha='center', va='bottom' if change > 0 else 'top',
                        fontsize=11, fontweight='bold')
            
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_ylabel('Change (%)', fontsize=12)
            ax2.set_title('Change Rate (Normal → Drowsy)', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(output_dir, f"blink_duration_comparison_{self.user_id}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 詳細グラフを保存しました: {output_path}")
        
        plt.show()
        
        return output_path


def main():
    """メイン関数"""
    print("=" * 60)
    print("   瞬き時間検証プログラム")
    print("   Blink Duration Verification Tool")
    print("=" * 60)
    
    # データディレクトリは固定
    data_dir = "data"
    
    # 検証クラスを初期化
    verifier = BlinkDurationVerifier(data_dir=data_dir)
    
    # 利用可能なユーザーを表示
    users = verifier.list_available_users()
    
    if not users:
        print("\n❌ 利用可能なユーザーが見つかりません")
        print("   データディレクトリを確認してください")
        return
    
    print(f"\n👥 利用可能なユーザーID ({len(users)}名):")
    for i, user in enumerate(users, 1):
        print(f"   {i}. {user}")
    
    # ユーザーIDを入力
    print("\n📝 検証するユーザーIDを入力してください")
    user_id = input("   > ").strip()
    
    if user_id not in users:
        print(f"\n⚠️ ユーザー '{user_id}' は見つかりません")
        print("   上記リストからユーザーIDを選択してください")
        return
    
    # データを読み込み
    success = verifier.load_user_data(user_id)
    
    if not success:
        print("\n❌ データの読み込みに失敗しました")
        return
    
    # 統計サマリーを表示
    verifier.print_statistics()
    
    # 統計検定を実施
    verifier.perform_statistical_tests()
    
    # グラフを作成
    print("\n📊 グラフを作成中...")
    verifier.create_comparison_plot()
    verifier.create_detailed_comparison_plot()
    
    print("\n✅ 検証完了!")


if __name__ == "__main__":
    main()