"""
楕円フィッティング対応 - BlinkDetector拡張版

EARに加えて、目の楕円パラメータ（長軸、短軸、面積、角度、偏心率）を抽出します。
"""

import numpy as np
import cv2
from collections import deque
import time


class BlinkDetectorWithEllipse:
    """
    EAR + 楕円フィッティングによる瞬き検出器
    
    抽出する瞬きパラメータ（拡張版）:
    1. 閉眼時間 (Tc)
    2. 開眼時間 (To)
    3. 瞬き係数 (To/Tc)
    4. 瞬き間隔
    5. EAR最小値
    6. 総瞬き時間
    7. 楕円長軸（最大時）
    8. 楕円短軸（最小時）
    9. 楕円面積（最小時）
    10. 楕円角度（最大変化時）
    11. 楕円偏心率（最大時）
    """
    
    def __init__(self):
        # ... (既存の初期化コード)
        
        # 楕円パラメータの履歴（瞬き中）
        self.current_blink_ellipse_params = []
        
        # 楕円フィッティングの設定
        self.min_points_for_ellipse = 5  # 楕円フィッティングに必要な最小点数
        
    def calculate_ellipse_parameters(self, eye_landmarks):
        """
        目のランドマークから楕円パラメータを計算
        
        Args:
            eye_landmarks: 目のランドマーク座標 [(x, y), ...]
            
        Returns:
            dict: 楕円パラメータ
                - major_axis: 長軸（横幅）
                - minor_axis: 短軸（縦幅）
                - area: 面積
                - angle: 角度（度）
                - eccentricity: 偏心率
                None: フィッティング失敗時
        """
        try:
            # ランドマーク数のチェック
            if len(eye_landmarks) < self.min_points_for_ellipse:
                return None
            
            # numpy配列に変換
            points = np.array(eye_landmarks, dtype=np.float32)
            
            # OpenCVの楕円フィッティング
            # ellipse = ((center_x, center_y), (width, height), angle)
            ellipse = cv2.fitEllipse(points)
            
            # パラメータの抽出
            center = ellipse[0]  # (x, y)
            axes = ellipse[1]    # (width, height)
            angle = ellipse[2]   # 角度（度）
            
            # 長軸・短軸の決定（widthとheightの大きい方が長軸）
            if axes[0] >= axes[1]:
                major_axis = axes[0]
                minor_axis = axes[1]
            else:
                major_axis = axes[1]
                minor_axis = axes[0]
            
            # 面積の計算
            area = np.pi * (major_axis / 2) * (minor_axis / 2)
            
            # 偏心率の計算
            # eccentricity = sqrt(1 - (b/a)^2)
            # b: 短軸, a: 長軸
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0.0
            
            return {
                'center': center,
                'major_axis': major_axis,
                'minor_axis': minor_axis,
                'area': area,
                'angle': angle,
                'eccentricity': eccentricity
            }
            
        except Exception as e:
            # フィッティング失敗
            return None
    
    def detect_blink_with_ellipse(self, ear, left_eye_landmarks, right_eye_landmarks):
        """
        EAR + 楕円パラメータによる瞬き検出
        
        Args:
            ear: Eye Aspect Ratio
            left_eye_landmarks: 左目のランドマーク
            right_eye_landmarks: 右目のランドマーク
            
        Returns:
            dict: 瞬き情報（拡張版）、瞬きが完了していない場合はNone
        """
        current_time = time.time()
        
        # 両目の楕円パラメータを計算
        left_ellipse = self.calculate_ellipse_parameters(left_eye_landmarks)
        right_ellipse = self.calculate_ellipse_parameters(right_eye_landmarks)
        
        # 楕円パラメータの平均を取る
        if left_ellipse and right_ellipse:
            avg_ellipse = {
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
        
        # 楕円パラメータを履歴に追加（瞬き中のみ）
        if self.blink_state in [self.BLINK_STATE_CLOSING, 
                                self.BLINK_STATE_CLOSED, 
                                self.BLINK_STATE_OPENING]:
            if avg_ellipse:
                self.current_blink_ellipse_params.append(avg_ellipse)
        
        # 既存の瞬き検出ロジック
        blink_info = self._detect_blink_state(ear, current_time)
        
        # 瞬き完了時、楕円パラメータを追加
        if blink_info is not None:
            # 楕円パラメータの統計を計算
            if len(self.current_blink_ellipse_params) > 0:
                ellipse_stats = self._calculate_ellipse_statistics(
                    self.current_blink_ellipse_params
                )
                blink_info.update(ellipse_stats)
            else:
                # 楕円パラメータが取れなかった場合はデフォルト値
                blink_info.update({
                    'ellipse_major_axis_max': 0.0,
                    'ellipse_minor_axis_min': 0.0,
                    'ellipse_area_min': 0.0,
                    'ellipse_angle_change': 0.0,
                    'ellipse_eccentricity_max': 0.0
                })
            
            # 楕円パラメータ履歴をクリア
            self.current_blink_ellipse_params = []
        
        return blink_info
    
    def _calculate_ellipse_statistics(self, ellipse_params_list):
        """
        瞬き中の楕円パラメータから統計量を計算
        
        Args:
            ellipse_params_list: 楕円パラメータのリスト
            
        Returns:
            dict: 楕円統計量
        """
        if len(ellipse_params_list) == 0:
            return {
                'ellipse_major_axis_max': 0.0,
                'ellipse_minor_axis_min': 0.0,
                'ellipse_area_min': 0.0,
                'ellipse_angle_change': 0.0,
                'ellipse_eccentricity_max': 0.0
            }
        
        # 各パラメータを抽出
        major_axes = [p['major_axis'] for p in ellipse_params_list]
        minor_axes = [p['minor_axis'] for p in ellipse_params_list]
        areas = [p['area'] for p in ellipse_params_list]
        angles = [p['angle'] for p in ellipse_params_list]
        eccentricities = [p['eccentricity'] for p in ellipse_params_list]
        
        # 統計量を計算
        stats = {
            'ellipse_major_axis_max': max(major_axes),           # 開眼時の最大横幅
            'ellipse_minor_axis_min': min(minor_axes),           # 閉眼時の最小縦幅
            'ellipse_area_min': min(areas),                      # 閉眼時の最小面積
            'ellipse_angle_change': max(angles) - min(angles),   # 角度の最大変化
            'ellipse_eccentricity_max': max(eccentricities)      # 最大偏心率
        }
        
        return stats
    
    def _detect_blink_state(self, ear, current_time):
        """
        既存の4段階瞬き検出ロジック
        （blink_detector_mediapipe.pyの既存コードをそのまま使用）
        """
        # ... (既存のコードをここに配置)
        pass


# 使用例
if __name__ == "__main__":
    detector = BlinkDetectorWithEllipse()
    
    # MediaPipeで取得したランドマーク
    left_eye = [(100, 100), (110, 95), (120, 100), (110, 105), (105, 102), (115, 102)]
    right_eye = [(200, 100), (210, 95), (220, 100), (210, 105), (205, 102), (215, 102)]
    
    # EAR計算
    ear = 0.25
    
    # 瞬き検出（楕円パラメータ込み）
    blink_info = detector.detect_blink_with_ellipse(ear, left_eye, right_eye)
    
    if blink_info:
        print("瞬き検出!")
        print(f"  閉眼時間: {blink_info['closing_time']*1000:.1f}ms")
        print(f"  開眼時間: {blink_info['opening_time']*1000:.1f}ms")
        print(f"  瞬き係数: {blink_info['blink_coefficient']:.2f}")
        print(f"  楕円長軸（最大）: {blink_info['ellipse_major_axis_max']:.1f}px")
        print(f"  楕円短軸（最小）: {blink_info['ellipse_minor_axis_min']:.1f}px")
        print(f"  楕円面積（最小）: {blink_info['ellipse_area_min']:.1f}px²")
        print(f"  楕円角度変化: {blink_info['ellipse_angle_change']:.1f}°")
        print(f"  楕円偏心率（最大）: {blink_info['ellipse_eccentricity_max']:.3f}")
