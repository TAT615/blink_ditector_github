"""
リアルタイム眠気推定システム（MediaPipe版 - 12次元特徴量対応）
Real-time Drowsiness Estimation System with MediaPipe - 12D Features

訓練済みLSTMモデル（12次元）を使用してリアルタイムで眠気を推定します。

使い方:
    python realtime_drowsiness_estimator_mediapipe_12d.py \
        --model-path trained_models/drowsiness_lstm_20251115_224046.pth
"""

import os
import sys
import cv2
import numpy as np
import time
import json
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Tuple
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
            tuple: (center_x, center_y, radius) または None
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
            
            return center_x, center_y, radius
            
        except:
            return None
    
    @staticmethod
    def fit_eyelids(eye_landmarks):
        """
        上まぶた・下まぶたの円をフィッティング
        
        Args:
            eye_landmarks: 目のランドマーク6点
                          [P1(目頭), P2(上), P3(上), P4(目尻), P5(下), P6(下)]
            
        Returns:
            tuple: ((c1_x, c1_y, c1_r), (c2_x, c2_y, c2_r)) または (None, None)
        """
        if len(eye_landmarks) < 6:
            return None, None
        
        # 上まぶた3点: P1(目頭), P2(上), P3(上)
        upper_points = [eye_landmarks[0], eye_landmarks[1], eye_landmarks[2]]
        c1 = TwoCircleFitter.fit_circle(upper_points)
        
        # 下まぶた3点: P1(目頭), P5(下), P6(下)
        lower_points = [eye_landmarks[0], eye_landmarks[4], eye_landmarks[5]]
        c2 = TwoCircleFitter.fit_circle(lower_points)
        
        return c1, c2


class DrowsinessLSTM(nn.Module):
    """眠気推定用LSTMモデル（12次元対応）"""
    
    def __init__(self, input_size=12, hidden_size1=64, hidden_size2=32, 
                 fc_size=32, num_classes=2, dropout_rate=0.3):
        super(DrowsinessLSTM, self).__init__()
        
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        # LSTM層
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 全結合層
        self.fc1 = nn.Linear(hidden_size2, fc_size)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(fc_size, num_classes)
    
    def forward(self, x):
        # x: (batch, sequence_length, input_size)
        
        # LSTM層1
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        # LSTM層2
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        # 最後の時刻の出力を使用
        last_output = lstm2_out[:, -1, :]
        
        # 全結合層
        fc1_out = self.fc1(last_output)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout3(fc1_out)
        
        output = self.fc2(fc1_out)
        
        return output


class RealtimeDrowsinessDetector:
    """リアルタイム眠気検知システム（12次元特徴量対応）"""
    
    # MediaPipe Face Meshのランドマークインデックス
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    # 瞬き状態
    STATE_OPEN = 0
    STATE_CLOSING = 1
    STATE_CLOSED = 2
    STATE_OPENING = 3
    
    def __init__(self, model_path, sequence_length=10, ear_threshold=0.21):
        """
        初期化
        
        Args:
            model_path (str): 学習済みモデルのパス
            sequence_length (int): LSTMのシーケンス長
            ear_threshold (float): EAR閾値
        """
        self.sequence_length = sequence_length
        self.ear_threshold = ear_threshold
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 瞬き検出用
        self.blink_state = self.STATE_OPEN
        self.state_start_time = time.time()
        self.t1 = 0  # OPEN → CLOSING
        self.t2 = 0  # CLOSING → CLOSED
        self.t3 = 0  # CLOSED → OPENING
        
        # 前回の瞬き時刻（間隔計算用）
        self.last_blink_time = None
        
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
        
        # 推定結果
        self.current_prediction = None
        self.current_probability = None
        
        # 統計
        self.total_blinks = 0
        self.drowsy_count = 0
        
        print("=" * 70)
        print("🚀 リアルタイム眠気検知システム起動")
        print("=" * 70)
        print(f"📁 モデル: {model_path}")
        print(f"📊 シーケンス長: {sequence_length}")
        print(f"👁️  EAR閾値: {ear_threshold}")
        print(f"🔢 特徴量次元: 12次元（2円方式）")
        print("=" * 70)
    
    def detect_blink(self, ear, left_eye_landmarks, right_eye_landmarks):
        """
        瞬きを検出し、完了時に12次元特徴量を抽出
        
        Args:
            ear (float): Eye Aspect Ratio
            left_eye_landmarks (list): 左目のランドマーク
            right_eye_landmarks (list): 右目のランドマーク
            
        Returns:
            np.array: 12次元特徴量 または None
        """
        current_time = time.time()
        
        # 状態遷移
        if self.blink_state == self.STATE_OPEN:
            if ear < self.ear_threshold:
                # OPEN → CLOSING
                self.blink_state = self.STATE_CLOSING
                self.t1 = current_time
                
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
                features = self.extract_blink_features(
                    left_eye_landmarks, 
                    right_eye_landmarks
                )
                
                if features is not None:
                    self.total_blinks += 1
                    self.last_blink_time = current_time
                    return features
        
        return None
    
    def extract_blink_features(self, left_eye_landmarks, right_eye_landmarks):
        """
        瞬き完了時に12次元特徴量を抽出
        
        12次元の内訳:
        1. closing_time (閉眼時間)
        2. opening_time (開眼時間)
        3. blink_coefficient (瞬き係数 To/Tc)
        4. timestamp (タイムスタンプ)
        5. total_duration (総瞬き時間)
        6. interval (瞬き間隔)
        7. c1_center_x (上まぶた円の中心X)
        8. c1_center_y (上まぶた円の中心Y)
        9. c1_radius (上まぶた円の半径)
        10. c2_center_x (下まぶた円の中心X)
        11. c2_center_y (下まぶた円の中心Y)
        12. c2_radius (下まぶた円の半径)
        
        Returns:
            np.array: 12次元特徴量 または None
        """
        try:
            current_time = time.time()
            
            # 時間パラメータの計算
            closing_time = self.t2 - self.t1
            opening_time = self.t3 - self.t2
            total_duration = closing_time + opening_time
            blink_coefficient = opening_time / closing_time if closing_time > 0 else 0
            
            # 有効性チェック
            if not (0.025 <= closing_time <= 1.0):
                return None
            if not (0.05 <= opening_time <= 0.6):
                return None
            if not (0.5 <= blink_coefficient <= 8.0):
                return None
            
            # 瞬き間隔の計算
            if self.last_blink_time is not None:
                interval = current_time - self.last_blink_time
            else:
                interval = 0.0
            
            # 2円パラメータを抽出
            c1_left, c2_left = TwoCircleFitter.fit_eyelids(left_eye_landmarks)
            c1_right, c2_right = TwoCircleFitter.fit_eyelids(right_eye_landmarks)
            
            # 両目の平均を取る
            if c1_left and c1_right and c2_left and c2_right:
                c1_center_x = (c1_left[0] + c1_right[0]) / 2.0
                c1_center_y = (c1_left[1] + c1_right[1]) / 2.0
                c1_radius = (c1_left[2] + c1_right[2]) / 2.0
                c2_center_x = (c2_left[0] + c2_right[0]) / 2.0
                c2_center_y = (c2_left[1] + c2_right[1]) / 2.0
                c2_radius = (c2_left[2] + c2_right[2]) / 2.0
            else:
                # 2円フィッティング失敗時はデフォルト値
                c1_center_x = c1_center_y = c1_radius = 0.0
                c2_center_x = c2_center_y = c2_radius = 0.0
            
            # 12次元特徴ベクトル
            features = np.array([
                closing_time,
                opening_time,
                blink_coefficient,
                self.t1,           # timestamp
                total_duration,
                interval,
                c1_center_x,
                c1_center_y,
                c1_radius,
                c2_center_x,
                c2_center_y,
                c2_radius
            ], dtype=np.float32)
            
            return features
            
        except Exception as e:
            print(f"⚠️  特徴量抽出エラー: {e}")
            return None
    
    def predict_drowsiness(self):
        """
        LSTMモデルで眠気を推定
        
        Returns:
            tuple: (予測クラス, 眠気確率) または (None, None)
        """
        if len(self.feature_buffer) < self.sequence_length:
            return None, None
        
        try:
            # シーケンスを作成
            sequence = np.array(list(self.feature_buffer))
            sequence = sequence.reshape(1, self.sequence_length, 12)
            
            # テンソルに変換
            sequence_tensor = torch.FloatTensor(sequence).to(self.device)
            
            # 推論
            with torch.no_grad():
                output = self.model(sequence_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                drowsy_prob = probabilities[0, 1].item()
            
            self.current_prediction = predicted_class
            self.current_probability = drowsy_prob
            
            if predicted_class == 1:
                self.drowsy_count += 1
            
            return predicted_class, drowsy_prob
            
        except Exception as e:
            print(f"⚠️  推論エラー: {e}")
            return None, None
    
    def process_frame(self, frame):
        """
        フレームを処理
        
        Args:
            frame: OpenCVのフレーム (BGR)
            
        Returns:
            tuple: (処理済みフレーム, EAR値, 予測クラス, 眠気確率)
        """
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
        
        # 瞬き検出
        blink_features = self.detect_blink(avg_ear, left_eye, right_eye)
        
        # 特徴量をバッファに追加
        if blink_features is not None:
            self.feature_buffer.append(blink_features)
        
        # 眠気推定
        pred_class, drowsy_prob = self.predict_drowsiness()
        
        # 目のランドマークを描画
        for point in left_eye + right_eye:
            cv2.circle(frame, point, 1, (0, 255, 0), -1)
        
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
        
        return frame, avg_ear, pred_class, drowsy_prob
    
    def run(self, camera_id=0):
        """
        リアルタイム推定を実行
        
        Args:
            camera_id (int): カメラID
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ カメラを開けませんでした")
            return
        
        print("\n🎥 カメラ起動")
        print("   'q' キーで終了")
        print("=" * 70)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # フレーム処理
                processed_frame, ear, pred_class, drowsy_prob = self.process_frame(frame)
                
                # 表示
                cv2.imshow('Drowsiness Detection (12D Features)', processed_frame)
                
                # キー入力待ち
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 統計表示
            print("\n" + "=" * 70)
            print("📊 セッション統計")
            print("=" * 70)
            print(f"総瞬き数: {self.total_blinks}")
            print(f"眠気検出回数: {self.drowsy_count}")
            print("=" * 70)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='リアルタイム眠気推定システム (12次元特徴量対応)')
    parser.add_argument('--model-path', type=str, required=True,
                        help='学習済みモデルのパス')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='LSTMのシーケンス長')
    parser.add_argument('--ear-threshold', type=float, default=0.21,
                        help='EAR閾値')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='カメラID')
    
    args = parser.parse_args()
    
    # 検出器を初期化
    detector = RealtimeDrowsinessDetector(
        model_path=args.model_path,
        sequence_length=args.sequence_length,
        ear_threshold=args.ear_threshold
    )
    
    # 実行
    detector.run(camera_id=args.camera_id)


if __name__ == "__main__":
    main()