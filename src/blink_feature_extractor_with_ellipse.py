"""
楕円パラメータ対応 - BlinkFeatureExtractor拡張版

従来の6次元特徴量に楕円パラメータ5次元を追加し、11次元の特徴ベクトルを生成します。
"""

import numpy as np
from collections import deque
import json


class BlinkFeatureExtractorWithEllipse:
    """
    瞬きデータから特徴量を抽出するクラス（楕円パラメータ対応）
    
    抽出する特徴量（11次元）:
    1. 瞬き係数 (To/Tc)
    2. 閉眼時間 Tc [秒]
    3. 開眼時間 To [秒]
    4. 瞬き間隔 [秒]
    5. EAR最小値
    6. 総瞬き時間 (Tc + To) [秒]
    7. 楕円長軸（最大）[ピクセル]
    8. 楕円短軸（最小）[ピクセル]
    9. 楕円面積（最小）[ピクセル²]
    10. 楕円角度変化 [度]
    11. 楕円偏心率（最大）
    """
    
    def __init__(self, sequence_length=10, buffer_size=100):
        """
        初期化
        
        Args:
            sequence_length (int): LSTM入力用のシーケンス長
            buffer_size (int): 特徴量履歴の最大保存数
        """
        self.sequence_length = sequence_length
        self.buffer_size = buffer_size
        
        # 瞬きデータの保存
        self.blink_features = deque(maxlen=buffer_size)
        self.raw_blink_data = deque(maxlen=buffer_size)
        
        # 前回の瞬き時刻
        self.last_blink_time = None
        
        # 正規化パラメータ
        self.normalization_params = {
            'mean': None,
            'std': None,
            'is_fitted': False
        }
        
        # 統計情報
        self.stats = {
            'total_blinks': 0,
            'valid_blinks': 0,
            'invalid_blinks': 0,
            'avg_blink_coefficient': [],
            'avg_closing_time': [],
            'avg_opening_time': [],
            'avg_ellipse_major_axis': [],
            'avg_ellipse_minor_axis': [],
            'avg_ellipse_area': []
        }
        
        # 異常値検出用の閾値（MediaPipe対応）
        self.validity_thresholds = {
            'min_tc': 0.025,     # 最小閉眼時間 [秒]
            'max_tc': 1.0,       # 最大閉眼時間 [秒]
            'min_to': 0.05,      # 最小開眼時間 [秒]
            'max_to': 0.6,       # 最大開眼時間 [秒]
            'min_interval': 0.1, # 最小瞬き間隔 [秒]
            'max_interval': 30.0,# 最大瞬き間隔 [秒]
            'min_ear': 0.0,      # 最小EAR値
            'max_ear': 0.5,      # 最大EAR値
            'min_coefficient': 0.5,  # 最小瞬き係数
            'max_coefficient': 8.0,  # 最大瞬き係数
            # 楕円パラメータの閾値
            'min_major_axis': 10.0,   # 最小長軸 [px]
            'max_major_axis': 100.0,  # 最大長軸 [px]
            'min_minor_axis': 2.0,    # 最小短軸 [px]
            'max_minor_axis': 50.0,   # 最大短軸 [px]
            'min_area': 20.0,         # 最小面積 [px²]
            'max_area': 3000.0,       # 最大面積 [px²]
            'max_angle_change': 45.0, # 最大角度変化 [度]
            'max_eccentricity': 1.0   # 最大偏心率
        }
    
    def extract_features(self, blink_data: dict) -> np.ndarray:
        """
        瞬きデータから11次元特徴ベクトルを抽出
        
        Args:
            blink_data (dict): 瞬きデータ
                - 't1': 閉じ始め時刻 [秒]
                - 't2': 完全閉眼時刻 [秒]
                - 't3': 開き終わり時刻 [秒]
                - 'ear_min': EAR最小値
                - 'ellipse_major_axis_max': 楕円長軸（最大）
                - 'ellipse_minor_axis_min': 楕円短軸（最小）
                - 'ellipse_area_min': 楕円面積（最小）
                - 'ellipse_angle_change': 楕円角度変化
                - 'ellipse_eccentricity_max': 楕円偏心率（最大）
                
        Returns:
            np.ndarray: 11次元特徴ベクトル、無効な場合はNone
        """
        try:
            # 時間パラメータの抽出
            t1 = blink_data.get('t1')
            t2 = blink_data.get('t2')
            t3 = blink_data.get('t3')
            ear_min = blink_data.get('ear_min', 0.0)
            
            # 楕円パラメータの抽出
            ellipse_major = blink_data.get('ellipse_major_axis_max', 0.0)
            ellipse_minor = blink_data.get('ellipse_minor_axis_min', 0.0)
            ellipse_area = blink_data.get('ellipse_area_min', 0.0)
            ellipse_angle = blink_data.get('ellipse_angle_change', 0.0)
            ellipse_ecc = blink_data.get('ellipse_eccentricity_max', 0.0)
            
            # 必須パラメータのチェック
            if t1 is None or t2 is None or t3 is None:
                print("⚠️ 必須パラメータが不足しています")
                self.stats['invalid_blinks'] += 1
                return None
            
            # 閉眼時間 Tc = T2 - T1
            tc = t2 - t1
            
            # 開眼時間 To = T3 - T2
            to = t3 - t2
            
            # 瞬き間隔の計算
            if self.last_blink_time is not None:
                blink_interval = t1 - self.last_blink_time
            else:
                blink_interval = 0.0
            
            # 総瞬き時間
            total_duration = tc + to
            
            # 瞬き係数 = To / Tc
            if tc > 0:
                blink_coefficient = to / tc
            else:
                print("⚠️ 閉眼時間が0以下です")
                self.stats['invalid_blinks'] += 1
                return None
            
            # データの妥当性チェック
            if not self._validate_features(tc, to, blink_interval, ear_min, 
                                           blink_coefficient,
                                           ellipse_major, ellipse_minor, 
                                           ellipse_area, ellipse_angle, ellipse_ecc):
                self.stats['invalid_blinks'] += 1
                return None
            
            # 11次元特徴ベクトルの作成
            features = np.array([
                blink_coefficient,    # 1. 瞬き係数
                tc,                   # 2. 閉眼時間
                to,                   # 3. 開眼時間
                blink_interval,       # 4. 瞬き間隔
                ear_min,              # 5. EAR最小値
                total_duration,       # 6. 総瞬き時間
                ellipse_major,        # 7. 楕円長軸（最大）
                ellipse_minor,        # 8. 楕円短軸（最小）
                ellipse_area,         # 9. 楕円面積（最小）
                ellipse_angle,        # 10. 楕円角度変化
                ellipse_ecc           # 11. 楕円偏心率（最大）
            ], dtype=np.float32)
            
            # 前回の瞬き時刻を更新
            self.last_blink_time = t1
            
            # 特徴量を保存
            self.blink_features.append(features)
            self.raw_blink_data.append(blink_data)
            
            # 統計情報の更新
            self.stats['total_blinks'] += 1
            self.stats['valid_blinks'] += 1
            self.stats['avg_blink_coefficient'].append(blink_coefficient)
            self.stats['avg_closing_time'].append(tc)
            self.stats['avg_opening_time'].append(to)
            self.stats['avg_ellipse_major_axis'].append(ellipse_major)
            self.stats['avg_ellipse_minor_axis'].append(ellipse_minor)
            self.stats['avg_ellipse_area'].append(ellipse_area)
            
            return features
            
        except Exception as e:
            print(f"❌ 特徴量抽出エラー: {e}")
            self.stats['invalid_blinks'] += 1
            return None
    
    def _validate_features(self, tc, to, interval, ear_min, coefficient,
                          ellipse_major, ellipse_minor, ellipse_area, 
                          ellipse_angle, ellipse_ecc):
        """
        特徴量の妥当性をチェック（楕円パラメータ含む）
        """
        # 既存のチェック（tc, to, interval, ear_min, coefficient）
        if not (self.validity_thresholds['min_tc'] <= tc <= self.validity_thresholds['max_tc']):
            print(f"⚠️ 閉眼時間が範囲外: {tc:.3f}秒")
            return False
        
        if not (self.validity_thresholds['min_to'] <= to <= self.validity_thresholds['max_to']):
            print(f"⚠️ 開眼時間が範囲外: {to:.3f}秒")
            return False
        
        if interval > 0:
            if not (self.validity_thresholds['min_interval'] <= interval <= self.validity_thresholds['max_interval']):
                print(f"⚠️ 瞬き間隔が範囲外: {interval:.3f}秒")
                return False
        
        if not (self.validity_thresholds['min_ear'] <= ear_min <= self.validity_thresholds['max_ear']):
            print(f"⚠️ EAR最小値が範囲外: {ear_min:.3f}")
            return False
        
        if not (self.validity_thresholds['min_coefficient'] <= coefficient <= self.validity_thresholds['max_coefficient']):
            print(f"⚠️ 瞬き係数が範囲外: {coefficient:.3f}")
            return False
        
        # 楕円パラメータのチェック（0の場合は検証をスキップ）
        if ellipse_major > 0:
            if not (self.validity_thresholds['min_major_axis'] <= ellipse_major <= self.validity_thresholds['max_major_axis']):
                print(f"⚠️ 楕円長軸が範囲外: {ellipse_major:.1f}px")
                return False
        
        if ellipse_minor > 0:
            if not (self.validity_thresholds['min_minor_axis'] <= ellipse_minor <= self.validity_thresholds['max_minor_axis']):
                print(f"⚠️ 楕円短軸が範囲外: {ellipse_minor:.1f}px")
                return False
        
        if ellipse_area > 0:
            if not (self.validity_thresholds['min_area'] <= ellipse_area <= self.validity_thresholds['max_area']):
                print(f"⚠️ 楕円面積が範囲外: {ellipse_area:.1f}px²")
                return False
        
        if abs(ellipse_angle) > self.validity_thresholds['max_angle_change']:
            print(f"⚠️ 楕円角度変化が範囲外: {ellipse_angle:.1f}°")
            return False
        
        if ellipse_ecc > self.validity_thresholds['max_eccentricity']:
            print(f"⚠️ 楕円偏心率が範囲外: {ellipse_ecc:.3f}")
            return False
        
        return True
    
    def get_sequence(self, normalize=True):
        """
        LSTM入力用のシーケンスデータを取得
        
        Returns:
            np.ndarray: shape (sequence_length, 11) のシーケンスデータ
        """
        if len(self.blink_features) < self.sequence_length:
            return None
        
        # 最新のsequence_length個の特徴量を取得
        recent_features = list(self.blink_features)[-self.sequence_length:]
        sequence = np.array(recent_features, dtype=np.float32)
        
        # 正規化
        if normalize and self.normalization_params['is_fitted']:
            sequence = self._normalize(sequence)
        
        return sequence
    
    def print_statistics(self):
        """統計情報を表示"""
        print("=" * 60)
        print("瞬き特徴量の統計情報（楕円パラメータ含む）")
        print("=" * 60)
        print(f"総瞬き数: {self.stats['total_blinks']}")
        print(f"有効な瞬き: {self.stats['valid_blinks']}")
        print(f"無効な瞬き: {self.stats['invalid_blinks']}")
        
        if len(self.stats['avg_blink_coefficient']) > 0:
            print(f"\n平均瞬き係数: {np.mean(self.stats['avg_blink_coefficient']):.2f}")
            print(f"平均閉眼時間: {np.mean(self.stats['avg_closing_time'])*1000:.1f}ms")
            print(f"平均開眼時間: {np.mean(self.stats['avg_opening_time'])*1000:.1f}ms")
            
        if len(self.stats['avg_ellipse_major_axis']) > 0:
            print(f"\n平均楕円長軸: {np.mean(self.stats['avg_ellipse_major_axis']):.1f}px")
            print(f"平均楕円短軸: {np.mean(self.stats['avg_ellipse_minor_axis']):.1f}px")
            print(f"平均楕円面積: {np.mean(self.stats['avg_ellipse_area']):.1f}px²")
        
        print("=" * 60)
