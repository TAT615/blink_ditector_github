"""
改善版: EARと楕円パラメータの時系列データを別々に保存

【重要な変更点】
1. 瞬き中のEAR時系列を保存
2. 瞬き中の楕円パラメータ時系列を保存
3. タイムスタンプ付きで詳細に記録
4. 統計量も計算して保存
"""

import numpy as np
import cv2
from collections import deque
import time


class BlinkDetectorWithTimeSeriesData:
    """
    EAR + 楕円フィッティングによる瞬き検出器（時系列データ保存版）
    
    保存するデータ:
    1. 瞬き統計量（従来通り）
    2. EAR時系列データ（新規）
    3. 楕円パラメータ時系列データ（新規）
    """
    
    # 瞬き状態の定義
    BLINK_STATE_OPEN = 0
    BLINK_STATE_CLOSING = 1
    BLINK_STATE_CLOSED = 2
    BLINK_STATE_OPENING = 3
    
    def __init__(self, ear_threshold=0.21):
        # 既存の初期化
        self.ear_threshold = ear_threshold
        self.blink_state = self.BLINK_STATE_OPEN
        
        # タイムスタンプ
        self.t1 = None  # 閉じ始め
        self.t2 = None  # 完全閉眼
        self.t3 = None  # 開き終わり
        
        # ===== 【新規】時系列データの保存 =====
        
        # EARの時系列（瞬き中のみ）
        self.current_blink_ear_timeseries = []
        
        # 楕円パラメータの時系列（瞬き中のみ）
        self.current_blink_ellipse_timeseries = []
        
        # 全瞬きの詳細データ履歴
        self.all_blinks_detailed_data = deque(maxlen=100)
        
        # 楕円フィッティングの設定
        self.min_points_for_ellipse = 5
    
    def calculate_ellipse_parameters(self, eye_landmarks):
        """
        目のランドマークから楕円パラメータを計算
        """
        try:
            if len(eye_landmarks) < self.min_points_for_ellipse:
                return None
            
            points = np.array(eye_landmarks, dtype=np.float32)
            ellipse = cv2.fitEllipse(points)
            
            center = ellipse[0]
            axes = ellipse[1]
            angle = ellipse[2]
            
            if axes[0] >= axes[1]:
                major_axis = axes[0]
                minor_axis = axes[1]
            else:
                major_axis = axes[1]
                minor_axis = axes[0]
            
            area = np.pi * (major_axis / 2) * (minor_axis / 2)
            
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0.0
            
            return {
                'center_x': center[0],
                'center_y': center[1],
                'major_axis': major_axis,
                'minor_axis': minor_axis,
                'area': area,
                'angle': angle,
                'eccentricity': eccentricity
            }
            
        except Exception as e:
            return None
    
    def detect_blink_with_timeseries(self, ear, left_eye_landmarks, right_eye_landmarks):
        """
        EAR + 楕円パラメータによる瞬き検出（時系列データ保存版）
        
        Args:
            ear: Eye Aspect Ratio
            left_eye_landmarks: 左目のランドマーク
            right_eye_landmarks: 右目のランドマーク
            
        Returns:
            dict: 瞬き情報（統計量 + 時系列データ）、瞬きが完了していない場合はNone
        """
        current_time = time.time()
        
        # 両目の楕円パラメータを計算
        left_ellipse = self.calculate_ellipse_parameters(left_eye_landmarks)
        right_ellipse = self.calculate_ellipse_parameters(right_eye_landmarks)
        
        # 楕円パラメータの平均
        if left_ellipse and right_ellipse:
            avg_ellipse = {
                'center_x': (left_ellipse['center_x'] + right_ellipse['center_x']) / 2,
                'center_y': (left_ellipse['center_y'] + right_ellipse['center_y']) / 2,
                'major_axis': (left_ellipse['major_axis'] + right_ellipse['major_axis']) / 2,
                'minor_axis': (left_ellipse['minor_axis'] + right_ellipse['minor_axis']) / 2,
                'area': (left_ellipse['area'] + right_ellipse['area']) / 2,
                'angle': (left_ellipse['angle'] + right_ellipse['angle']) / 2,
                'eccentricity': (left_ellipse['eccentricity'] + right_ellipse['eccentricity']) / 2
            }
        elif left_ellipse:
            avg_ellipse = left_ellipse
        elif right_ellipse:
            avg_ellipse = right_ellipse
        else:
            avg_ellipse = None
        
        # ===== 【重要】時系列データの保存 =====
        if self.blink_state in [self.BLINK_STATE_CLOSING, 
                                self.BLINK_STATE_CLOSED, 
                                self.BLINK_STATE_OPENING]:
            
            # EARを時系列データとして保存
            self.current_blink_ear_timeseries.append({
                'timestamp': current_time,
                'ear': ear,
                'state': self._get_state_name(self.blink_state)
            })
            
            # 楕円パラメータを時系列データとして保存
            if avg_ellipse:
                ellipse_data = avg_ellipse.copy()
                ellipse_data['timestamp'] = current_time
                ellipse_data['state'] = self._get_state_name(self.blink_state)
                self.current_blink_ellipse_timeseries.append(ellipse_data)
        
        # 瞬き検出ロジック
        blink_info = self._detect_blink_state(ear, current_time)
        
        # 瞬き完了時の処理
        if blink_info is not None:
            # 統計量の計算
            if len(self.current_blink_ellipse_timeseries) > 0:
                ellipse_stats = self._calculate_ellipse_statistics(
                    self.current_blink_ellipse_timeseries
                )
                blink_info.update(ellipse_stats)
            else:
                blink_info.update({
                    'ellipse_major_axis_max': 0.0,
                    'ellipse_minor_axis_min': 0.0,
                    'ellipse_area_min': 0.0,
                    'ellipse_angle_change': 0.0,
                    'ellipse_eccentricity_max': 0.0
                })
            
            # ===== 【重要】時系列データを blink_info に追加 =====
            blink_info['ear_timeseries'] = self.current_blink_ear_timeseries.copy()
            blink_info['ellipse_timeseries'] = self.current_blink_ellipse_timeseries.copy()
            
            # 詳細データを履歴に保存
            self.all_blinks_detailed_data.append(blink_info.copy())
            
            # 時系列データをクリア
            self.current_blink_ear_timeseries = []
            self.current_blink_ellipse_timeseries = []
        
        return blink_info
    
    def _get_state_name(self, state):
        """状態名を取得"""
        state_names = {
            self.BLINK_STATE_OPEN: "OPEN",
            self.BLINK_STATE_CLOSING: "CLOSING",
            self.BLINK_STATE_CLOSED: "CLOSED",
            self.BLINK_STATE_OPENING: "OPENING"
        }
        return state_names.get(state, "UNKNOWN")
    
    def _calculate_ellipse_statistics(self, ellipse_timeseries):
        """楕円パラメータの時系列から統計量を計算"""
        if len(ellipse_timeseries) == 0:
            return {
                'ellipse_major_axis_max': 0.0,
                'ellipse_minor_axis_min': 0.0,
                'ellipse_area_min': 0.0,
                'ellipse_angle_change': 0.0,
                'ellipse_eccentricity_max': 0.0
            }
        
        major_axes = [e['major_axis'] for e in ellipse_timeseries]
        minor_axes = [e['minor_axis'] for e in ellipse_timeseries]
        areas = [e['area'] for e in ellipse_timeseries]
        angles = [e['angle'] for e in ellipse_timeseries]
        eccentricities = [e['eccentricity'] for e in ellipse_timeseries]
        
        return {
            'ellipse_major_axis_max': max(major_axes),
            'ellipse_minor_axis_min': min(minor_axes),
            'ellipse_area_min': min(areas),
            'ellipse_angle_change': max(angles) - min(angles),
            'ellipse_eccentricity_max': max(eccentricities)
        }
    
    def _detect_blink_state(self, ear, current_time):
        """
        4段階の瞬き検出
        （既存のロジックをそのまま使用）
        """
        # OPEN → CLOSING
        if self.blink_state == self.BLINK_STATE_OPEN:
            if ear < self.ear_threshold:
                self.blink_state = self.BLINK_STATE_CLOSING
                self.t1 = current_time
        
        # CLOSING → CLOSED
        elif self.blink_state == self.BLINK_STATE_CLOSING:
            # EARが継続して閾値以下なら CLOSED へ
            if ear < self.ear_threshold:
                self.blink_state = self.BLINK_STATE_CLOSED
                self.t2 = current_time
            # EARが閾値を超えたらキャンセル
            else:
                self.blink_state = self.BLINK_STATE_OPEN
                self.t1 = None
        
        # CLOSED → OPENING
        elif self.blink_state == self.BLINK_STATE_CLOSED:
            if ear >= self.ear_threshold:
                self.blink_state = self.BLINK_STATE_OPENING
        
        # OPENING → OPEN（瞬き完了）
        elif self.blink_state == self.BLINK_STATE_OPENING:
            if ear >= self.ear_threshold:
                self.t3 = current_time
                
                # 瞬き情報を作成
                if self.t1 and self.t2 and self.t3:
                    tc = self.t2 - self.t1
                    to = self.t3 - self.t2
                    
                    blink_info = {
                        't1': self.t1,
                        't2': self.t2,
                        't3': self.t3,
                        'closing_time': tc,
                        'opening_time': to,
                        'blink_coefficient': to / tc if tc > 0 else 0,
                        'total_duration': tc + to
                    }
                    
                    # 状態をリセット
                    self.blink_state = self.BLINK_STATE_OPEN
                    self.t1 = None
                    self.t2 = None
                    self.t3 = None
                    
                    return blink_info
        
        return None
    
    def get_all_detailed_data(self):
        """
        すべての瞬きの詳細データを取得
        
        Returns:
            list: 各瞬きの詳細データ（統計量 + 時系列）
        """
        return list(self.all_blinks_detailed_data)


# ===== 使用例 =====
if __name__ == "__main__":
    detector = BlinkDetectorWithTimeSeriesData()
    
    # シミュレーション用のデータ
    left_eye = [(100, 100), (110, 95), (120, 100), (110, 105), (105, 102), (115, 102)]
    right_eye = [(200, 100), (210, 95), (220, 100), (210, 105), (205, 102), (215, 102)]
    
    # 瞬きのシミュレーション
    ear_sequence = [0.30, 0.25, 0.20, 0.18, 0.20, 0.25, 0.30]
    
    for i, ear in enumerate(ear_sequence):
        blink_info = detector.detect_blink_with_timeseries(ear, left_eye, right_eye)
        
        if blink_info:
            print("\n" + "=" * 60)
            print("瞬き検出!")
            print("=" * 60)
            
            # 統計量
            print("\n【統計量】")
            print(f"  閉眼時間: {blink_info['closing_time']*1000:.1f}ms")
            print(f"  開眼時間: {blink_info['opening_time']*1000:.1f}ms")
            print(f"  瞬き係数: {blink_info['blink_coefficient']:.2f}")
            print(f"  楕円長軸（最大）: {blink_info['ellipse_major_axis_max']:.1f}px")
            print(f"  楕円短軸（最小）: {blink_info['ellipse_minor_axis_min']:.1f}px")
            
            # EAR時系列
            print("\n【EAR時系列データ】")
            for ear_data in blink_info['ear_timeseries']:
                print(f"  {ear_data['state']:8s} | EAR: {ear_data['ear']:.3f}")
            
            # 楕円時系列
            print("\n【楕円時系列データ】")
            for ellipse_data in blink_info['ellipse_timeseries']:
                print(f"  {ellipse_data['state']:8s} | "
                      f"長軸: {ellipse_data['major_axis']:5.1f}px | "
                      f"短軸: {ellipse_data['minor_axis']:5.1f}px | "
                      f"面積: {ellipse_data['area']:6.1f}px²")
