"""
リアルタイム眠気推定システム（MediaPipe版 - 12次元特徴量完全対応・結果保存機能付き）
Real-time Drowsiness Estimation System with MediaPipe - Full 12D Features with Result Saving

訓練済みLSTMモデル（12次元）を使用してリアルタイムで眠気を推定します。
EAR手法と2つの円手法を統合した完全版。

特徴量構成（12次元）:
    [0] closing_time: 閉眼時間
    [1] opening_time: 開眼時間
    [2] blink_coefficient: 瞬き係数 (opening_time / closing_time)
    [3] interval: 前回の瞬きからの間隔
    [4] total_duration: 総持続時間
    [5] upper_radius_max: 上まぶた円の最大半径
    [6] lower_radius_max: 下まぶた円の最大半径
    [7] vertical_distance_min: 上下円の最小距離
    [8] radius_diff_max: 半径差の最大値
    [9] eye_height_min: 目の高さの最小値
    [10] eye_width_avg: 目の幅の平均値
    [11] ear_min: EARの最小値

使い方:
    python -m src.realtime_drowsiness_estimator_mediapipe \
        --model-path trained_models/drowsiness_lstm_20251125_004040.pth
"""

import os
import sys
import cv2
import numpy as np
import time
import json
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Tuple, List
import argparse
import mediapipe as mp

# PyTorchのインポート
try:
    import torch
    import torch.nn as nn
except ImportError as e:
    print(f"❌ PyTorchのインポートエラー: {e}")
    print("   pip install torch でインストールしてください")
    sys.exit(1)


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
        v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        
        # 水平距離
        h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        if h == 0:
            return 0.0
        
        # EAR計算
        ear = (v1 + v2) / (2.0 * h)
        return ear


class TwoCircleFitter:
    """2円フィッティングクラス（上まぶた・下まぶた）"""
    
    @staticmethod
    def fit_circle(points):
        """
        3点から円をフィッティング
        
        Args:
            points: [(x, y), (x, y), (x, y)]
            
        Returns:
            dict: 円のパラメータ {center_x, center_y, radius} または None
        """
        if len(points) != 3:
            return None
        
        try:
            points = np.array(points, dtype=np.float32)
            
            # 行列Aとベクトルbを構築
            A = np.zeros((3, 3))
            b = np.zeros(3)
            
            for i in range(3):
                x, y = points[i]
                A[i] = [2*x, 2*y, 1]
                b[i] = x*x + y*y
            
            # 連立方程式を解く
            params = np.linalg.solve(A, b)
            
            center_x = params[0]
            center_y = params[1]
            radius = np.sqrt(params[2] + center_x**2 + center_y**2)
            
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
                          [P0(左端), P1(上左), P2(上右), P3(右端), P4(下右), P5(下左)]
            
        Returns:
            dict: 2円パラメータ {
                'upper_circle': {center_x, center_y, radius},
                'lower_circle': {center_x, center_y, radius},
                'vertical_distance': 上下の円の中心間距離,
                'radius_diff': 半径の差,
                'eye_height': 目の高さ,
                'eye_width': 目の幅
            } または None
        """
        if len(eye_landmarks) < 6:
            return None
        
        try:
            # 上まぶた3点: P1(上左), P2(上右), P3(右端)
            upper_points = [eye_landmarks[1], eye_landmarks[2], eye_landmarks[3]]
            upper_circle = TwoCircleFitter.fit_circle(upper_points)
            
            if upper_circle is None:
                return None
            
            # 下まぶた3点: P0(左端), P4(下右), P5(下左)
            lower_points = [eye_landmarks[0], eye_landmarks[4], eye_landmarks[5]]
            lower_circle = TwoCircleFitter.fit_circle(lower_points)
            
            if lower_circle is None:
                return None
            
            # 2円の中心間距離（垂直距離）
            vertical_distance = np.sqrt(
                (upper_circle['center_x'] - lower_circle['center_x'])**2 +
                (upper_circle['center_y'] - lower_circle['center_y'])**2
            )
            
            # 半径の差
            radius_diff = abs(upper_circle['radius'] - lower_circle['radius'])
            
            # 目の高さ（上下の垂直距離の平均）
            eye_height = (
                np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5])) +
                np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
            ) / 2.0
            
            # 目の幅（水平距離）
            eye_width = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
            
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


class DrowsinessLSTM(nn.Module):
    """眠気推定用LSTMモデル（12次元特徴量対応）"""
    
    def __init__(self, input_size=12, hidden_size1=64, hidden_size2=32, 
                 fc_size=32, num_classes=2, dropout_rate=0.3):
        super(DrowsinessLSTM, self).__init__()
        
        # 2層LSTM
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            batch_first=True,
            dropout=dropout_rate if hidden_size2 > 0 else 0
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size2,
            batch_first=True
        )
        
        # Dropout層
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # 全結合層
        self.fc1 = nn.Linear(hidden_size2, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # LSTM1
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        # LSTM2
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        # 最後の時刻の出力
        last_output = lstm2_out[:, -1, :]
        
        # 全結合層
        fc1_out = self.fc1(last_output)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout3(fc1_out)
        
        output = self.fc2(fc1_out)
        
        return output


class RealtimeDrowsinessDetector:
    """リアルタイム眠気検知システム（12次元特徴量完全対応・結果保存機能付き）"""
    
    # MediaPipe Face Meshのランドマークインデックス
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    # 瞬き状態
    STATE_OPEN = 0
    STATE_CLOSING = 1
    STATE_CLOSED = 2
    STATE_OPENING = 3
    
    def __init__(self, model_path, sequence_length=10, ear_threshold=0.21,
                 output_dir="drowsiness_results"):
        """
        初期化
        
        Args:
            model_path (str): 学習済みモデルのパス
            sequence_length (int): LSTMのシーケンス長
            ear_threshold (float): EAR閾値
            output_dir (str): 結果保存ディレクトリ
        """
        self.sequence_length = sequence_length
        self.ear_threshold = ear_threshold
        self.output_dir = output_dir
        self.model_path = model_path
        
        # 結果保存ディレクトリを作成
        os.makedirs(output_dir, exist_ok=True)
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 瞬き検出用変数
        self.blink_state = self.STATE_OPEN
        self.state_start_time = time.time()
        self.t1 = 0  # OPEN → CLOSING
        self.t2 = 0  # CLOSING → CLOSED
        self.t3 = 0  # CLOSED → OPENING
        
        # 前回の瞬き時刻（間隔計算用）
        self.last_blink_time = None
        
        # 2円パラメータの最小値追跡（瞬き中）
        self.current_blink_circles_data = []
        
        # EAR履歴（瞬き中のEAR最小値を記録）
        self.current_blink_ear_history = []
        
        # 特徴量バッファ（12次元 × sequence_length）
        self.feature_buffer = deque(maxlen=sequence_length)
        
        # LSTMモデルの読み込み
        print("🧠 モデル読み込み中...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model_params = checkpoint.get('model_params', {})
        
        self.model = DrowsinessLSTM(
            input_size=model_params.get('input_size', 12),
            hidden_size1=model_params.get('hidden_size1', 64),
            hidden_size2=model_params.get('hidden_size2', 32),
            fc_size=model_params.get('fc_size', 32),
            num_classes=model_params.get('num_classes', 2),
            dropout_rate=model_params.get('dropout_rate', 0.3)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"   デバイス: {self.device}")
        print(f"   入力次元: {model_params.get('input_size', 12)}")
        print(f"✅ モデルを読み込みました: {model_path}")
        print(f"📁 結果保存先: {output_dir}")
        
        # 推定結果
        self.current_prediction = None
        self.current_probability = None
        
        # 統計
        self.total_blinks = 0
        self.drowsy_count = 0
        self.normal_count = 0
        self.frame_count = 0
        
        # アラート管理
        self.consecutive_drowsy = 0
        self.consecutive_drowsy_threshold = 3
        self.alert_active = False
        
        # セッション情報（結果保存用）
        self.session_start_time = datetime.now()
        self.blink_history = []
        self.prediction_history = []
        self.ear_samples = []
        
        print("=" * 70)
        print("🚀 リアルタイム眠気検知システム起動（結果保存機能付き）")
        print("=" * 70)
        print(f"📁 モデル: {model_path}")
        print(f"📊 シーケンス長: {sequence_length}")
        print(f"👁️  EAR閾値: {ear_threshold}")
        print(f"🔢 特徴量次元: 12次元（Temporal + Spatial）")
        print(f"💾 結果保存先: {output_dir}")
        print("=" * 70)
    
    def detect_blink(self, ear, left_eye_landmarks, right_eye_landmarks):
        """
        瞬きを検出し、完了時に12次元特徴量を抽出
        
        Args:
            ear (float): Eye Aspect Ratio
            left_eye_landmarks (list): 左目のランドマーク6点
            right_eye_landmarks (list): 右目のランドマーク6点
            
        Returns:
            np.array: 12次元特徴量 または None
        """
        current_time = time.time()
        
        # 瞬き中は2円パラメータとEARを記録
        if self.blink_state in [self.STATE_CLOSING, self.STATE_CLOSED, self.STATE_OPENING]:
            # EAR履歴を記録
            self.current_blink_ear_history.append(ear)
            
            # 両目の2円パラメータ
            left_circles = TwoCircleFitter.fit_eyelids(left_eye_landmarks)
            right_circles = TwoCircleFitter.fit_eyelids(right_eye_landmarks)
            
            # 平均値を計算して記録
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
                self.current_blink_circles_data.append(avg_circles)
        
        # 状態遷移
        if self.blink_state == self.STATE_OPEN:
            if ear < self.ear_threshold:
                # OPEN → CLOSING
                self.blink_state = self.STATE_CLOSING
                self.t1 = current_time
                self.current_blink_circles_data = []
                self.current_blink_ear_history = []
                
        elif self.blink_state == self.STATE_CLOSING:
            if ear < self.ear_threshold * 0.8:
                # CLOSING → CLOSED
                self.blink_state = self.STATE_CLOSED
                self.t2 = current_time
                
        elif self.blink_state == self.STATE_CLOSED:
            if ear > self.ear_threshold:
                # CLOSED → OPENING
                self.blink_state = self.STATE_OPENING
                self.t3 = current_time
                
        elif self.blink_state == self.STATE_OPENING:
            if ear > self.ear_threshold * 1.2:
                # OPENING → OPEN (瞬き完了)
                self.blink_state = self.STATE_OPEN
                
                # 12次元特徴量を抽出
                features = self.extract_blink_features_12d()
                
                # 統計更新
                self.total_blinks += 1
                
                # リセット
                self.t1 = 0
                self.t2 = 0
                self.t3 = 0
                self.current_blink_circles_data = []
                self.current_blink_ear_history = []
                
                return features
        
        return None
    
    def extract_blink_features_12d(self):
        """
        瞬き完了時に12次元特徴量を抽出
        
        Returns:
            np.array: 12次元特徴量ベクトル
                [0] closing_time: 閉眼時間
                [1] opening_time: 開眼時間
                [2] blink_coefficient: 瞬き係数 (opening_time / closing_time)
                [3] interval: 前回の瞬きからの間隔
                [4] total_duration: 総持続時間
                [5] upper_radius_max: 上まぶた円の最大半径
                [6] lower_radius_max: 下まぶた円の最大半径
                [7] vertical_distance_min: 上下円の最小距離
                [8] radius_diff_max: 半径差の最大値
                [9] eye_height_min: 目の高さの最小値
                [10] eye_width_avg: 目の幅の平均値
                [11] ear_min: EARの最小値
        """
        # Temporal特徴量（時間的パラメータ）
        closing_time = self.t2 - self.t1 if self.t1 and self.t2 else 0.0
        opening_time = self.t3 - self.t2 if self.t2 and self.t3 else 0.0
        blink_coefficient = opening_time / closing_time if closing_time > 0 else 0.0
        total_duration = closing_time + opening_time
        
        # 瞬き間隔
        current_time = time.time()
        interval = current_time - self.last_blink_time if self.last_blink_time else 0.0
        self.last_blink_time = current_time
        
        # Spatial特徴量（空間的パラメータ）- 2円方式
        if len(self.current_blink_circles_data) > 0:
            # 各パラメータの統計値を計算
            upper_radii = [d['upper_radius'] for d in self.current_blink_circles_data]
            lower_radii = [d['lower_radius'] for d in self.current_blink_circles_data]
            vertical_distances = [d['vertical_distance'] for d in self.current_blink_circles_data]
            radius_diffs = [d['radius_diff'] for d in self.current_blink_circles_data]
            eye_heights = [d['eye_height'] for d in self.current_blink_circles_data]
            eye_widths = [d['eye_width'] for d in self.current_blink_circles_data]
            
            upper_radius_max = max(upper_radii) if upper_radii else 0.0
            lower_radius_max = max(lower_radii) if lower_radii else 0.0
            vertical_distance_min = min(vertical_distances) if vertical_distances else 0.0
            radius_diff_max = max(radius_diffs) if radius_diffs else 0.0
            eye_height_min = min(eye_heights) if eye_heights else 0.0
            eye_width_avg = np.mean(eye_widths) if eye_widths else 0.0
        else:
            # デフォルト値
            upper_radius_max = 0.0
            lower_radius_max = 0.0
            vertical_distance_min = 0.0
            radius_diff_max = 0.0
            eye_height_min = 0.0
            eye_width_avg = 0.0
        
        # EARの最小値
        if len(self.current_blink_ear_history) > 0:
            ear_min = min(self.current_blink_ear_history)
        else:
            ear_min = 0.0
        
        # 12次元特徴量ベクトル
        features = np.array([
            closing_time,           # [0]
            opening_time,           # [1]
            blink_coefficient,      # [2]
            interval,               # [3]
            total_duration,         # [4]
            upper_radius_max,       # [5]
            lower_radius_max,       # [6]
            vertical_distance_min,  # [7]
            radius_diff_max,        # [8]
            eye_height_min,         # [9]
            eye_width_avg,          # [10]
            ear_min                 # [11]
        ], dtype=np.float32)
        
        # 瞬き履歴に追加（結果保存用）
        self.blink_history.append({
            'timestamp': datetime.now().isoformat(),
            'blink_number': self.total_blinks + 1,
            'closing_time_ms': closing_time * 1000,
            'opening_time_ms': opening_time * 1000,
            'total_duration_ms': total_duration * 1000,
            'blink_coefficient': blink_coefficient,
            'interval_s': interval,
            'upper_radius_max': upper_radius_max,
            'lower_radius_max': lower_radius_max,
            'vertical_distance_min': vertical_distance_min,
            'radius_diff_max': radius_diff_max,
            'eye_height_min': eye_height_min,
            'eye_width_avg': eye_width_avg,
            'ear_min': ear_min
        })
        
        return features
    
    def predict_drowsiness(self):
        """
        眠気を推定
        
        Returns:
            tuple: (予測クラス, 眠気確率)
        """
        if len(self.feature_buffer) < self.sequence_length:
            return None, None
        
        # バッファから最新のシーケンスを取得
        sequence = np.array(list(self.feature_buffer), dtype=np.float32)
        sequence = sequence.reshape(1, self.sequence_length, -1)
        
        # PyTorchテンソルに変換
        sequence_tensor = torch.FloatTensor(sequence).to(self.device)
        
        # 推論
        with torch.no_grad():
            output = self.model(sequence_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            drowsy_prob = probabilities[0, 1].item()
        
        # 統計更新
        if pred_class == 1:
            self.drowsy_count += 1
            self.consecutive_drowsy += 1
        else:
            self.normal_count += 1
            self.consecutive_drowsy = 0
        
        # アラート判定
        if self.consecutive_drowsy >= self.consecutive_drowsy_threshold:
            self.alert_active = True
        else:
            self.alert_active = False
        
        # 推定履歴に追加（結果保存用）
        self.prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'predicted_class': pred_class,
            'drowsy_probability': drowsy_prob,
            'blink_count_at_prediction': self.total_blinks
        })
        
        return pred_class, drowsy_prob
    
    def process_frame(self, frame):
        """
        フレームを処理
        
        Args:
            frame: 入力フレーム
            
        Returns:
            tuple: (処理済みフレーム, EAR値, 予測クラス, 眠気確率)
        """
        self.frame_count += 1
        
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
        
        # EARサンプリング（100フレームごと）
        if self.frame_count % 100 == 0:
            self.ear_samples.append({
                'timestamp': datetime.now().isoformat(),
                'frame': self.frame_count,
                'ear': avg_ear
            })
        
        # 瞬き検出と特徴量抽出
        blink_features = self.detect_blink(avg_ear, left_eye, right_eye)
        
        # 特徴量をバッファに追加
        if blink_features is not None:
            self.feature_buffer.append(blink_features)
        
        # 眠気推定
        pred_class, drowsy_prob = self.predict_drowsiness()
        
        # 目のランドマークを描画
        for point in left_eye:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        for point in right_eye:
            cv2.circle(frame, point, 2, (255, 0, 0), -1)
        
        # 2円の描画（視覚化）
        left_circles = TwoCircleFitter.fit_eyelids(left_eye)
        right_circles = TwoCircleFitter.fit_eyelids(right_eye)
        
        # 左目の2円を描画
        if left_circles:
            try:
                # 上まぶた円（シアン色）
                upper_center = (int(left_circles['upper_circle']['center_x']),
                              int(left_circles['upper_circle']['center_y']))
                upper_radius = int(left_circles['upper_circle']['radius'])
                cv2.circle(frame, upper_center, upper_radius, (255, 255, 0), 2)
                
                # 下まぶた円（黄色）
                lower_center = (int(left_circles['lower_circle']['center_x']),
                              int(left_circles['lower_circle']['center_y']))
                lower_radius = int(left_circles['lower_circle']['radius'])
                cv2.circle(frame, lower_center, lower_radius, (0, 255, 255), 2)
            except:
                pass
        
        # 右目の2円を描画
        if right_circles:
            try:
                # 上まぶた円（シアン色）
                upper_center = (int(right_circles['upper_circle']['center_x']),
                              int(right_circles['upper_circle']['center_y']))
                upper_radius = int(right_circles['upper_circle']['radius'])
                cv2.circle(frame, upper_center, upper_radius, (255, 255, 0), 2)
                
                # 下まぶた円（黄色）
                lower_center = (int(right_circles['lower_circle']['center_x']),
                              int(right_circles['lower_circle']['center_y']))
                lower_radius = int(right_circles['lower_circle']['radius'])
                cv2.circle(frame, lower_center, lower_radius, (0, 255, 255), 2)
            except:
                pass
        
        # 情報表示
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Buffer: {len(self.feature_buffer)}/{self.sequence_length}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if pred_class is not None:
            status = "DROWSY" if pred_class == 1 else "NORMAL"
            color = (0, 0, 255) if pred_class == 1 else (0, 255, 0)
            
            cv2.putText(frame, f"Status: {status}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(frame, f"Drowsy Prob: {drowsy_prob:.2%}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # アラート表示
            if self.alert_active:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
                cv2.putText(frame, "!!! ALERT: DROWSY !!!", (w//4, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        return frame, avg_ear, pred_class, drowsy_prob
    
    def get_statistics(self):
        """統計情報を取得"""
        total_predictions = self.drowsy_count + self.normal_count
        
        return {
            'total_blinks': self.total_blinks,
            'total_predictions': total_predictions,
            'drowsy_count': self.drowsy_count,
            'normal_count': self.normal_count,
            'drowsy_percentage': (self.drowsy_count / total_predictions * 100) if total_predictions > 0 else 0,
            'alert_active': self.alert_active
        }
    
    def save_results(self):
        """セッション結果をJSONファイルに保存"""
        session_end_time = datetime.now()
        duration = (session_end_time - self.session_start_time).total_seconds()
        
        # 統計情報の計算
        total_predictions = self.drowsy_count + self.normal_count
        
        # 瞬き統計
        blink_stats = {}
        if len(self.blink_history) > 0:
            closing_times = [b['closing_time_ms'] for b in self.blink_history]
            opening_times = [b['opening_time_ms'] for b in self.blink_history]
            total_durations = [b['total_duration_ms'] for b in self.blink_history]
            coefficients = [b['blink_coefficient'] for b in self.blink_history]
            intervals = [b['interval_s'] for b in self.blink_history if b['interval_s'] > 0]
            
            blink_stats = {
                'closing_time_ms': {
                    'mean': np.mean(closing_times),
                    'std': np.std(closing_times),
                    'min': np.min(closing_times),
                    'max': np.max(closing_times)
                },
                'opening_time_ms': {
                    'mean': np.mean(opening_times),
                    'std': np.std(opening_times),
                    'min': np.min(opening_times),
                    'max': np.max(opening_times)
                },
                'total_duration_ms': {
                    'mean': np.mean(total_durations),
                    'std': np.std(total_durations),
                    'min': np.min(total_durations),
                    'max': np.max(total_durations)
                },
                'blink_coefficient': {
                    'mean': np.mean(coefficients),
                    'std': np.std(coefficients),
                    'min': np.min(coefficients),
                    'max': np.max(coefficients)
                },
                'interval_s': {
                    'mean': np.mean(intervals) if intervals else 0,
                    'std': np.std(intervals) if intervals else 0,
                    'min': np.min(intervals) if intervals else 0,
                    'max': np.max(intervals) if intervals else 0
                }
            }
        
        # 結果データ
        result = {
            'session_info': {
                'start_time': self.session_start_time.isoformat(),
                'end_time': session_end_time.isoformat(),
                'duration_seconds': duration,
                'model_path': self.model_path,
                'ear_threshold': self.ear_threshold,
                'sequence_length': self.sequence_length
            },
            'statistics': {
                'total_frames': self.frame_count,
                'total_blinks': self.total_blinks,
                'total_predictions': total_predictions,
                'normal_predictions': self.normal_count,
                'drowsy_predictions': self.drowsy_count,
                'drowsy_ratio': self.drowsy_count / total_predictions if total_predictions > 0 else 0,
                'blinks_per_minute': self.total_blinks / (duration / 60) if duration > 0 else 0,
                'average_blink_interval': np.mean([b['interval_s'] for b in self.blink_history if b['interval_s'] > 0]) if self.blink_history else 0
            },
            'blink_statistics': blink_stats,
            'blink_history': self.blink_history,
            'prediction_history': self.prediction_history,
            'ear_samples': self.ear_samples
        }
        
        # ファイル名を生成
        filename = f"session_{self.session_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # JSONファイルに保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 結果を保存しました: {filepath}")
        
        return filepath
    
    def run(self, camera_id=0):
        """
        メインループを実行
        
        Args:
            camera_id: カメラID
        """
        # カメラ初期化
        print("\n🎥 カメラ起動")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ カメラを開けませんでした")
            return
        
        # カメラ設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("   'q' キーで終了")
        print("   's' キーで途中保存")
        print("   'r' キーで統計リセット")
        print("=" * 70)
        
        # FPS計測用
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ フレームの取得に失敗しました")
                    break
                
                # フレーム処理
                processed_frame, ear, pred_class, drowsy_prob = self.process_frame(frame)
                
                # FPS計算
                fps_frame_count += 1
                if fps_frame_count >= 30:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_frame_count = 0
                
                # FPS表示
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # フレーム表示
                cv2.imshow('Drowsiness Detection (12D Features)', processed_frame)
                
                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('s'):  # s - 途中保存
                    self.save_results()
                elif key == ord('r'):  # r - リセット
                    self.total_blinks = 0
                    self.drowsy_count = 0
                    self.normal_count = 0
                    self.consecutive_drowsy = 0
                    self.alert_active = False
                    self.blink_history = []
                    self.prediction_history = []
                    print("\n✅ 統計をリセットしました\n")
        
        except KeyboardInterrupt:
            print("\n\n⚠️ ユーザーによって中断されました")
        
        finally:
            # 最終統計表示
            stats = self.get_statistics()
            print("\n" + "=" * 70)
            print("📊 セッション統計")
            print("=" * 70)
            print(f"総瞬き数: {stats['total_blinks']}")
            print(f"眠気検出回数: {stats['drowsy_count']}")
            print(f"正常検出回数: {stats['normal_count']}")
            print(f"眠気割合: {stats['drowsy_percentage']:.1f}%")
            print("=" * 70)
            
            # 結果を保存
            filepath = self.save_results()
            print(f"\n✅ セッション結果を保存しました: {filepath}")
            
            # クリーンアップ
            cap.release()
            cv2.destroyAllWindows()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='リアルタイム眠気推定システム (12次元特徴量・結果保存機能付き)')
    parser.add_argument('--model-path', type=str, required=True,
                        help='学習済みモデルのパス')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='LSTMのシーケンス長')
    parser.add_argument('--ear-threshold', type=float, default=0.21,
                        help='EAR閾値')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='カメラID')
    parser.add_argument('--output-dir', type=str, default='drowsiness_results',
                        help='結果保存ディレクトリ')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 リアルタイム眠気検知システム起動（12次元・結果保存機能付き）")
    print("=" * 70)
    print(f"📁 モデル: {args.model_path}")
    print(f"📊 シーケンス長: {args.sequence_length}")
    print(f"👁️ EAR閾値: {args.ear_threshold}")
    print(f"💾 結果保存先: {args.output_dir}")
    print("=" * 70)
    
    detector = RealtimeDrowsinessDetector(
        model_path=args.model_path,
        sequence_length=args.sequence_length,
        ear_threshold=args.ear_threshold,
        output_dir=args.output_dir
    )
    
    detector.run(camera_id=args.camera_id)


if __name__ == "__main__":
    main()
