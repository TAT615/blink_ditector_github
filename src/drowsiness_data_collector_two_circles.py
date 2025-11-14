"""
眠気検出システム - データ収集プログラム（2円方式：上まぶた・下まぶた）

上まぶたと下まぶたを別々の円で検出します（C1, C2方式）
- MediaPipe Face Meshで顔・目を検出
- 上まぶた3点から円C1をフィッティング
- 下まぶた3点から円C2をフィッティング
- EAR（Eye Aspect Ratio）を計算
- 瞬き検出（4段階）
- KSS（Karolinska Sleepiness Scale）眠気アンケート
- 統計量 + 時系列データをJSONに保存

使い方:
    # プログラムを実行
    python drowsiness_data_collector_two_circles.py
    
    # 対話的に以下を入力:
    # 1. ユーザーID（3桁の番号: 001-999）
    # 2. 状態（1: 正常状態、2: 眠気状態）
    # 3. KSS眠気スコア（1-10）
    # 4. Enterキーで記録開始
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
import argparse
from datetime import datetime
from collections import deque


class EARCalculator:
    """Eye Aspect Ratio（EAR）計算クラス"""
    
    @staticmethod
    def calculate(eye_landmarks):
        """
        EARを計算
        
        Args:
            eye_landmarks: 目のランドマーク6点 [(x,y), ...]
            
        Returns:
            float: EAR値
        """
        if len(eye_landmarks) != 6:
            return 0.0
        
        # 垂直距離
        A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        
        # 水平距離
        C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        if C == 0:
            return 0.0
        
        # EAR = (A + B) / (2 * C)
        ear = (A + B) / (2.0 * C)
        
        return ear


class TwoCircleFitter:
    """2円フィッティングクラス（上まぶた・下まぶた）"""
    
    @staticmethod
    def fit_circle(points):
        """
        3点以上から円をフィッティング
        
        Args:
            points: [(x, y), ...] 3点以上
            
        Returns:
            dict: 円のパラメータ {center_x, center_y, radius}
        """
        try:
            if len(points) < 3:
                return None
            
            points_array = np.array(points, dtype=np.float32)
            
            # 最小二乗法で円をフィッティング
            # (x - cx)^2 + (y - cy)^2 = r^2
            # x^2 + y^2 = 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2)
            
            n = len(points_array)
            x = points_array[:, 0]
            y = points_array[:, 1]
            
            # 行列Aとベクトルbを構築
            A = np.column_stack([2*x, 2*y, np.ones(n)])
            b = x**2 + y**2
            
            # 最小二乗法で解く
            params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            center_x = params[0]
            center_y = params[1]
            c = params[2]
            
            radius = np.sqrt(c + center_x**2 + center_y**2)
            
            return {
                'center_x': float(center_x),
                'center_y': float(center_y),
                'radius': float(radius)
            }
            
        except Exception as e:
            return None
    
    @staticmethod
    def fit_eyelids(eye_landmarks):
        """
        目のランドマークから上まぶた円と下まぶた円をフィッティング
        
        Args:
            eye_landmarks: 目のランドマーク6点
                          [P1(目頭), P2(上), P3(上), P4(目尻), P5(下), P6(下)]
            
        Returns:
            dict: {
                'upper_circle': 上まぶた円のパラメータ,
                'lower_circle': 下まぶた円のパラメータ,
                'vertical_distance': 垂直距離（円の中心間）
            }
        """
        try:
            if len(eye_landmarks) != 6:
                return None
            
            # 上まぶた3点: 目頭、上2点、目尻
            upper_points = [
                eye_landmarks[0],  # P1: 目頭
                eye_landmarks[1],  # P2: 上
                eye_landmarks[2],  # P3: 上
                eye_landmarks[3]   # P4: 目尻
            ]
            
            # 下まぶた3点: 目頭、下2点、目尻
            lower_points = [
                eye_landmarks[0],  # P1: 目頭
                eye_landmarks[5],  # P6: 下
                eye_landmarks[4],  # P5: 下
                eye_landmarks[3]   # P4: 目尻
            ]
            
            # 円をフィッティング
            upper_circle = TwoCircleFitter.fit_circle(upper_points)
            lower_circle = TwoCircleFitter.fit_circle(lower_points)
            
            if upper_circle is None or lower_circle is None:
                return None
            
            # 垂直距離を計算
            vertical_distance = abs(upper_circle['center_y'] - lower_circle['center_y'])
            
            # 半径の差
            radius_diff = abs(upper_circle['radius'] - lower_circle['radius'])
            
            # 目の高さ（近似値）
            eye_height = vertical_distance
            
            # 目の幅（2つの円の平均半径から推定）
            avg_radius = (upper_circle['radius'] + lower_circle['radius']) / 2
            eye_width = avg_radius * 2
            
            return {
                'upper_circle': upper_circle,
                'lower_circle': lower_circle,
                'vertical_distance': float(vertical_distance),
                'radius_diff': float(radius_diff),
                'eye_height': float(eye_height),
                'eye_width': float(eye_width)
            }
            
        except Exception as e:
            return None


class BlinkDetectorTwoCircles:
    """
    瞬き検出器（2円方式対応）
    """
    
    # 瞬き状態の定義
    BLINK_STATE_OPEN = 0
    BLINK_STATE_CLOSING = 1
    BLINK_STATE_CLOSED = 2
    BLINK_STATE_OPENING = 3
    
    def __init__(self, ear_threshold=0.21):
        self.ear_threshold = ear_threshold
        self.blink_state = self.BLINK_STATE_OPEN
        
        # タイムスタンプ
        self.t1 = None  # 閉じ始め
        self.t2 = None  # 完全閉眼
        self.t3 = None  # 開き終わり
        
        # 時系列データの保存
        self.current_blink_ear_timeseries = []
        self.current_blink_circles_timeseries = []
        
        # EAR履歴（統計用）
        self.ear_history = []
        
        # 前回の瞬き時刻
        self.last_blink_time = None
    
    def _get_state_name(self):
        """状態名を取得"""
        state_names = {
            self.BLINK_STATE_OPEN: "OPEN",
            self.BLINK_STATE_CLOSING: "CLOSING",
            self.BLINK_STATE_CLOSED: "CLOSED",
            self.BLINK_STATE_OPENING: "OPENING"
        }
        return state_names.get(self.blink_state, "UNKNOWN")
    
    def detect(self, ear, left_eye_landmarks, right_eye_landmarks):
        """
        瞬きを検出
        
        Args:
            ear: Eye Aspect Ratio
            left_eye_landmarks: 左目のランドマーク
            right_eye_landmarks: 右目のランドマーク
            
        Returns:
            dict: 瞬き情報（完了時のみ）、None（瞬き中または未検出）
        """
        current_time = time.time()
        
        # 両目の2円パラメータを計算
        left_circles = TwoCircleFitter.fit_eyelids(left_eye_landmarks)
        right_circles = TwoCircleFitter.fit_eyelids(right_eye_landmarks)
        
        # 平均値を計算
        if left_circles and right_circles:
            avg_circles = {
                'upper_radius': (left_circles['upper_circle']['radius'] + 
                               right_circles['upper_circle']['radius']) / 2,
                'lower_radius': (left_circles['lower_circle']['radius'] + 
                               right_circles['lower_circle']['radius']) / 2,
                'vertical_distance': (left_circles['vertical_distance'] + 
                                    right_circles['vertical_distance']) / 2,
                'radius_diff': (left_circles['radius_diff'] + 
                              right_circles['radius_diff']) / 2,
                'eye_height': (left_circles['eye_height'] + 
                             right_circles['eye_height']) / 2,
                'eye_width': (left_circles['eye_width'] + 
                            right_circles['eye_width']) / 2
            }
        elif left_circles:
            avg_circles = {
                'upper_radius': left_circles['upper_circle']['radius'],
                'lower_radius': left_circles['lower_circle']['radius'],
                'vertical_distance': left_circles['vertical_distance'],
                'radius_diff': left_circles['radius_diff'],
                'eye_height': left_circles['eye_height'],
                'eye_width': left_circles['eye_width']
            }
        elif right_circles:
            avg_circles = {
                'upper_radius': right_circles['upper_circle']['radius'],
                'lower_radius': right_circles['lower_circle']['radius'],
                'vertical_distance': right_circles['vertical_distance'],
                'radius_diff': right_circles['radius_diff'],
                'eye_height': right_circles['eye_height'],
                'eye_width': right_circles['eye_width']
            }
        else:
            avg_circles = None
        
        # 時系列データの保存（瞬き中のみ）
        if self.blink_state in [self.BLINK_STATE_CLOSING, 
                                self.BLINK_STATE_CLOSED, 
                                self.BLINK_STATE_OPENING]:
            
            # EAR時系列
            self.current_blink_ear_timeseries.append({
                'timestamp': current_time,
                'ear': float(ear),
                'state': self._get_state_name()
            })
            
            # EAR履歴
            self.ear_history.append(ear)
            
            # 2円時系列
            if avg_circles:
                circles_data = avg_circles.copy()
                circles_data['timestamp'] = current_time
                circles_data['state'] = self._get_state_name()
                self.current_blink_circles_timeseries.append(circles_data)
        
        # 瞬き検出ロジック
        blink_info = self._detect_blink_state(ear, current_time)
        
        # 瞬き完了時の処理
        if blink_info is not None:
            # 2円統計量の計算
            if len(self.current_blink_circles_timeseries) > 0:
                circles_stats = self._calculate_circles_statistics()
                blink_info.update(circles_stats)
            else:
                blink_info.update({
                    'upper_radius_max': 0.0,
                    'lower_radius_max': 0.0,
                    'vertical_distance_min': 0.0,
                    'radius_diff_max': 0.0,
                    'eye_height_min': 0.0,
                    'eye_width_avg': 0.0
                })
            
            # EAR最小値
            if len(self.ear_history) > 0:
                blink_info['ear_min'] = float(min(self.ear_history))
            else:
                blink_info['ear_min'] = float(ear)
            
            # 瞬き間隔の計算
            if self.last_blink_time is not None:
                blink_info['interval'] = float(self.t1 - self.last_blink_time)
            else:
                blink_info['interval'] = 0.0
            
            self.last_blink_time = self.t1
            
            # 時系列データを追加
            blink_info['ear_timeseries'] = self.current_blink_ear_timeseries.copy()
            blink_info['circles_timeseries'] = self.current_blink_circles_timeseries.copy()
            
            # クリア
            self.current_blink_ear_timeseries = []
            self.current_blink_circles_timeseries = []
            self.ear_history = []
        
        return blink_info
    
    def _calculate_circles_statistics(self):
        """2円パラメータの統計量を計算"""
        if len(self.current_blink_circles_timeseries) == 0:
            return {
                'upper_radius_max': 0.0,
                'lower_radius_max': 0.0,
                'vertical_distance_min': 0.0,
                'radius_diff_max': 0.0,
                'eye_height_min': 0.0,
                'eye_width_avg': 0.0
            }
        
        upper_radii = [c['upper_radius'] for c in self.current_blink_circles_timeseries]
        lower_radii = [c['lower_radius'] for c in self.current_blink_circles_timeseries]
        vert_distances = [c['vertical_distance'] for c in self.current_blink_circles_timeseries]
        radius_diffs = [c['radius_diff'] for c in self.current_blink_circles_timeseries]
        eye_heights = [c['eye_height'] for c in self.current_blink_circles_timeseries]
        eye_widths = [c['eye_width'] for c in self.current_blink_circles_timeseries]
        
        return {
            'upper_radius_max': float(max(upper_radii)),
            'lower_radius_max': float(max(lower_radii)),
            'vertical_distance_min': float(min(vert_distances)),
            'radius_diff_max': float(max(radius_diffs)),
            'eye_height_min': float(min(eye_heights)),
            'eye_width_avg': float(np.mean(eye_widths))
        }
    
    def _detect_blink_state(self, ear, current_time):
        """4段階の瞬き検出"""
        
        # OPEN → CLOSING
        if self.blink_state == self.BLINK_STATE_OPEN:
            if ear < self.ear_threshold:
                self.blink_state = self.BLINK_STATE_CLOSING
                self.t1 = current_time
                self.ear_history = [ear]
        
        # CLOSING → CLOSED
        elif self.blink_state == self.BLINK_STATE_CLOSING:
            if ear < self.ear_threshold:
                self.blink_state = self.BLINK_STATE_CLOSED
                self.t2 = current_time
            else:
                # キャンセル
                self.blink_state = self.BLINK_STATE_OPEN
                self.t1 = None
                self.ear_history = []
        
        # CLOSED → OPENING
        elif self.blink_state == self.BLINK_STATE_CLOSED:
            if ear >= self.ear_threshold:
                self.blink_state = self.BLINK_STATE_OPENING
        
        # OPENING → OPEN（瞬き完了）
        elif self.blink_state == self.BLINK_STATE_OPENING:
            if ear >= self.ear_threshold:
                self.t3 = current_time
                
                if self.t1 and self.t2 and self.t3:
                    tc = self.t2 - self.t1
                    to = self.t3 - self.t2
                    
                    blink_info = {
                        't1': float(self.t1),
                        't2': float(self.t2),
                        't3': float(self.t3),
                        'closing_time': float(tc),
                        'opening_time': float(to),
                        'blink_coefficient': float(to / tc) if tc > 0 else 0.0,
                        'total_duration': float(tc + to)
                    }
                    
                    # 状態をリセット
                    self.blink_state = self.BLINK_STATE_OPEN
                    self.t1 = None
                    self.t2 = None
                    self.t3 = None
                    
                    return blink_info
        
        return None


class DataCollectorTwoCircles:
    """データ収集クラス（2円方式）"""
    
    def __init__(self, output_dir="data/sessions"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 目のランドマークインデックス
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        
        # 瞬き検出器
        self.blink_detector = BlinkDetectorTwoCircles()
        
        # セッション情報
        self.session_data = {
            'session_id': None,
            'user_id': None,
            'label': None,
            'kss_score': None,
            'start_time': None,
            'end_time': None,
            'duration': 0.0,
            'total_blinks': 0,
            'valid_blinks': 0,
            'blinks': []
        }
        
        self.blink_counter = 0
        self.session_start_time = None
    
    def start_session(self, user_id, label, kss_score):
        """セッション開始"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_str = "normal" if label == 0 else "drowsy"
        
        self.session_data['session_id'] = f"{timestamp}_{user_id}_{label_str}"
        self.session_data['user_id'] = user_id
        self.session_data['label'] = label
        self.session_data['kss_score'] = kss_score
        self.session_data['start_time'] = datetime.now().isoformat()
        self.session_data['blinks'] = []
        self.blink_counter = 0
        self.session_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"セッション開始")
        print(f"{'='*60}")
        print(f"セッションID: {self.session_data['session_id']}")
        print(f"ユーザーID: {user_id}")
        print(f"状態: {label_str}")
        print(f"KSS眠気スコア: {kss_score}")
        print(f"検出方式: 2円方式（上まぶた・下まぶた）")
        print(f"{'='*60}\n")
    
    def add_blink(self, blink_info):
        """瞬きデータを追加"""
        if blink_info is None:
            return
        
        self.blink_counter += 1
        
        blink_data = {
            'blink_id': self.blink_counter,
            'timestamp': blink_info['t1'],
            'statistics': {
                'closing_time': blink_info['closing_time'],
                'opening_time': blink_info['opening_time'],
                'blink_coefficient': blink_info['blink_coefficient'],
                'total_duration': blink_info['total_duration'],
                'interval': blink_info['interval'],
                'ear_min': blink_info['ear_min'],
                # 2円パラメータ
                'upper_radius_max': blink_info['upper_radius_max'],
                'lower_radius_max': blink_info['lower_radius_max'],
                'vertical_distance_min': blink_info['vertical_distance_min'],
                'radius_diff_max': blink_info['radius_diff_max'],
                'eye_height_min': blink_info['eye_height_min'],
                'eye_width_avg': blink_info['eye_width_avg']
            },
            'ear_timeseries': blink_info['ear_timeseries'],
            'circles_timeseries': blink_info['circles_timeseries']
        }
        
        self.session_data['blinks'].append(blink_data)
        self.session_data['total_blinks'] += 1
        
        # 簡易的な有効性チェック
        if self._is_valid_blink(blink_data['statistics']):
            self.session_data['valid_blinks'] += 1
        
        print(f"瞬き検出 #{self.blink_counter}: "
              f"係数={blink_info['blink_coefficient']:.2f}, "
              f"Tc={blink_info['closing_time']*1000:.0f}ms, "
              f"To={blink_info['opening_time']*1000:.0f}ms, "
              f"垂直距離={blink_info['vertical_distance_min']:.1f}px")
    
    def _is_valid_blink(self, stats):
        """瞬きの有効性チェック（簡易版）"""
        tc = stats['closing_time']
        to = stats['opening_time']
        coef = stats['blink_coefficient']
        
        if not (0.025 <= tc <= 1.0):
            return False
        if not (0.05 <= to <= 0.6):
            return False
        if not (0.5 <= coef <= 8.0):
            return False
        
        return True
    
    def end_session(self):
        """セッション終了してJSONファイルに保存"""
        self.session_data['end_time'] = datetime.now().isoformat()
        self.session_data['duration'] = time.time() - self.session_start_time
        
        # JSONファイルに保存
        filepath = os.path.join(
            self.output_dir,
            f"{self.session_data['session_id']}.json"
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"セッション終了: {self.session_data['session_id']}")
        print(f"総瞬き数: {self.session_data['total_blinks']}")
        print(f"有効な瞬き: {self.session_data['valid_blinks']}")
        print(f"保存先: {filepath}")
        print(f"{'='*60}\n")
        
        return filepath
    
    def process_frame(self, frame):
        """フレームを処理"""
        # RGB変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipeで顔検出
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return frame, None, None, None
        
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # 目のランドマークを取得
        left_eye = [(int(face_landmarks.landmark[i].x * w),
                     int(face_landmarks.landmark[i].y * h))
                    for i in self.LEFT_EYE_INDICES]
        
        right_eye = [(int(face_landmarks.landmark[i].x * w),
                      int(face_landmarks.landmark[i].y * h))
                     for i in self.RIGHT_EYE_INDICES]
        
        # EAR計算
        left_ear = EARCalculator.calculate(left_eye)
        right_ear = EARCalculator.calculate(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # 2円フィッティング
        left_circles = TwoCircleFitter.fit_eyelids(left_eye)
        right_circles = TwoCircleFitter.fit_eyelids(right_eye)
        
        # 平均値を計算
        if left_circles and right_circles:
            avg_circles = {
                'upper_radius': (left_circles['upper_circle']['radius'] + 
                               right_circles['upper_circle']['radius']) / 2,
                'lower_radius': (left_circles['lower_circle']['radius'] + 
                               right_circles['lower_circle']['radius']) / 2,
                'vertical_distance': (left_circles['vertical_distance'] + 
                                    right_circles['vertical_distance']) / 2
            }
        elif left_circles:
            avg_circles = {
                'upper_radius': left_circles['upper_circle']['radius'],
                'lower_radius': left_circles['lower_circle']['radius'],
                'vertical_distance': left_circles['vertical_distance']
            }
        elif right_circles:
            avg_circles = {
                'upper_radius': right_circles['upper_circle']['radius'],
                'lower_radius': right_circles['lower_circle']['radius'],
                'vertical_distance': right_circles['vertical_distance']
            }
        else:
            avg_circles = None
        
        # 瞬き検出
        blink_info = self.blink_detector.detect(avg_ear, left_eye, right_eye)
        
        # 可視化
        frame = self._draw_visualization(frame, left_eye, right_eye, 
                                        left_circles, right_circles,
                                        avg_ear, avg_circles)
        
        return frame, blink_info, avg_ear, avg_circles
    
    def _draw_visualization(self, frame, left_eye, right_eye, 
                           left_circles, right_circles, avg_ear, avg_circles):
        """可視化"""
        # 目のランドマークを描画
        for point in left_eye + right_eye:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        
        # 左目の2円を描画
        if left_circles:
            try:
                # 上まぶた円（赤色）
                upper_center = (int(left_circles['upper_circle']['center_x']),
                              int(left_circles['upper_circle']['center_y']))
                upper_radius = int(left_circles['upper_circle']['radius'])
                cv2.circle(frame, upper_center, upper_radius, (0, 0, 255), 2)
                
                # 下まぶた円（青色）
                lower_center = (int(left_circles['lower_circle']['center_x']),
                              int(left_circles['lower_circle']['center_y']))
                lower_radius = int(left_circles['lower_circle']['radius'])
                cv2.circle(frame, lower_center, lower_radius, (255, 0, 0), 2)
            except:
                pass
        
        # 右目の2円を描画
        if right_circles:
            try:
                # 上まぶた円（赤色）
                upper_center = (int(right_circles['upper_circle']['center_x']),
                              int(right_circles['upper_circle']['center_y']))
                upper_radius = int(right_circles['upper_circle']['radius'])
                cv2.circle(frame, upper_center, upper_radius, (0, 0, 255), 2)
                
                # 下まぶた円（青色）
                lower_center = (int(right_circles['lower_circle']['center_x']),
                              int(right_circles['lower_circle']['center_y']))
                lower_radius = int(right_circles['lower_circle']['radius'])
                cv2.circle(frame, lower_center, lower_radius, (255, 0, 0), 2)
            except:
                pass
        
        # 情報表示
        y_offset = 30
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if avg_circles:
            y_offset += 30
            cv2.putText(frame, f"Upper R: {avg_circles['upper_radius']:.1f}px",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            y_offset += 25
            cv2.putText(frame, f"Lower R: {avg_circles['lower_radius']:.1f}px",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            y_offset += 25
            cv2.putText(frame, f"Vert Dist: {avg_circles['vertical_distance']:.1f}px",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 瞬き状態
        y_offset += 30
        state_name = self.blink_detector._get_state_name()
        cv2.putText(frame, f"State: {state_name}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 瞬きカウント
        y_offset += 30
        cv2.putText(frame, f"Blinks: {self.blink_counter}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return frame


def get_user_input():
    """ユーザーから対話的に入力を取得"""
    print("\n" + "="*60)
    print("眠気検出データ収集プログラム（2円方式）")
    print("="*60 + "\n")
    
    # ユーザーIDの入力
    while True:
        user_id = input("ユーザーIDを入力してください（3桁の番号: 001-999）\n> ").strip()
        
        # バリデーション
        if len(user_id) == 3 and user_id.isdigit():
            print(f"✓ ユーザーID: {user_id}\n")
            break
        else:
            print("❌ エラー: 3桁の数字を入力してください（例: 001, 123, 999）\n")
    
    # 状態の選択
    while True:
        print("状態を選択してください:")
        print("  1. 正常状態（Normal）")
        print("  2. 眠気状態（Drowsy）")
        choice = input("選択 (1 or 2): ").strip()
        
        if choice == '1':
            label = 0
            label_str = "正常状態（Normal）"
            print(f"✓ 状態: {label_str}\n")
            break
        elif choice == '2':
            label = 1
            label_str = "眠気状態（Drowsy）"
            print(f"✓ 状態: {label_str}\n")
            break
        else:
            print("❌ エラー: 1 または 2 を入力してください\n")
    
    # KSS眠気アンケート
    print("="*60)
    print("KSS眠気アンケート (Karolinska Sleepiness Scale)")
    print("="*60)
    print("現在のあなたの眠気レベルを選択してください:\n")
    print("  1  = 非常に覚醒している")
    print("  2  = とても覚醒している")
    print("  3  = 覚醒している")
    print("  4  = やや覚醒している")
    print("  5  = 覚醒も眠気もない")
    print("  6  = 眠気の兆候がある")
    print("  7  = 眠いが、覚醒を保つのに苦労はない")
    print("  8  = 眠く、覚醒を保つのに少し努力が必要")
    print("  9  = 非常に眠く、覚醒を保つのに大変な努力が必要")
    print("  10 = 極度に眠く、起きていられない\n")
    
    while True:
        kss_input = input("眠気レベル (1-10): ").strip()
        
        # バリデーション
        if kss_input.isdigit() and 1 <= int(kss_input) <= 10:
            kss_score = int(kss_input)
            print(f"✓ KSSスコア: {kss_score}\n")
            break
        else:
            print("❌ エラー: 1 から 10 の数字を入力してください\n")
    
    # 確認画面
    print("="*60)
    print("=== 確認 ===")
    print(f"ユーザーID: {user_id}")
    print(f"状態: {label_str}")
    print(f"KSS眠気スコア: {kss_score}")
    print("="*60)
    input("\nEnterキーを押すと記録を開始します...")
    
    return user_id, label, kss_score


def main():
    """メイン関数"""
    # 対話的に入力を取得
    user_id, label, kss_score = get_user_input()
    
    # データ収集器の初期化
    output_dir = 'data/sessions'
    collector = DataCollectorTwoCircles(output_dir=output_dir)
    
    # セッション開始
    collector.start_session(user_id=user_id, label=label, kss_score=kss_score)
    
    # カメラ初期化
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ カメラを開けませんでした")
        return
    
    print("📹 カメラ起動成功")
    print("\n操作方法:")
    print("  ESC: セッション終了・保存")
    print("  Q: セッション終了・保存")
    print("\n瞬きを自然に行ってください...\n")
    print("画面表示:")
    print("  赤い円: 上まぶた（C1）")
    print("  青い円: 下まぶた（C2）")
    print("  緑の点: 目のランドマーク\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ フレームの取得に失敗しました")
                break
            
            # フレーム処理
            processed_frame, blink_info, avg_ear, avg_circles = collector.process_frame(frame)
            
            # 瞬き検出時
            if blink_info is not None:
                collector.add_blink(blink_info)
            
            # 画面表示
            cv2.imshow('Data Collection (Two Circles)', processed_frame)
            
            # キー入力
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or Q
                break
    
    except KeyboardInterrupt:
        print("\n⚠️ 中断されました")
    
    finally:
        # セッション終了
        filepath = collector.end_session()
        
        # クリーンアップ
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ データ収集完了: {filepath}")


if __name__ == "__main__":
    main()
