"""
リアルタイム眠気検知システム（2円方式・12次元特徴量対応）

カメラで顔を撮影し、瞬きパターンから眠気状態をリアルタイムで判定します。

使い方:
    python realtime_drowsiness_estimator.py --model-path trained_models/drowsiness_lstm_20251115_224046.pth
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import argparse
import time
from collections import deque
from datetime import datetime

# LSTMモデルのインポート
try:
    from src.lstm_drowsiness_model import DrowsinessEstimator, DrowsinessLSTM
except ImportError:
    print("⚠️ src/lstm_drowsiness_model.py が見つかりません")
    print("   プロジェクトルートから実行してください")


class EARCalculator:
    """Eye Aspect Ratio（EAR）計算器"""
    
    @staticmethod
    def calculate(eye_landmarks):
        """
        EARを計算
        
        Args:
            eye_landmarks: 目のランドマーク座標リスト [(x, y), ...]
            
        Returns:
            float: EAR値
        """
        # 垂直距離
        v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        
        # 水平距離
        h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        # EAR計算
        ear = (v1 + v2) / (2.0 * h)
        return ear


class TwoCircleFitter:
    """2円フィッティング（上まぶた・下まぶた）"""
    
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
            eye_landmarks: 目のランドマーク座標リスト
            
        Returns:
            tuple: ((c1_x, c1_y, c1_r), (c2_x, c2_y, c2_r)) または (None, None)
        """
        if len(eye_landmarks) < 6:
            return None, None
        
        # 上まぶた3点
        upper_points = [eye_landmarks[1], eye_landmarks[2], eye_landmarks[5]]
        c1 = TwoCircleFitter.fit_circle(upper_points)
        
        # 下まぶた3点
        lower_points = [eye_landmarks[3], eye_landmarks[4], eye_landmarks[5]]
        c2 = TwoCircleFitter.fit_circle(lower_points)
        
        return c1, c2


class RealtimeDrowsinessDetector:
    """
    リアルタイム眠気検知システム
    """
    
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
        
        # 特徴量バッファ（シーケンス用）
        self.feature_buffer = deque(maxlen=sequence_length)
        
        # LSTMモデルの読み込み
        self.estimator = DrowsinessEstimator()
        self.estimator.load_model(model_path)
        self.estimator.model.eval()
        
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
        print(f"👁️ EAR閾値: {ear_threshold}")
        print("=" * 70)
    
    def extract_eye_landmarks(self, face_landmarks, image_width, image_height, eye_indices):
        """
        目のランドマークを抽出
        
        Args:
            face_landmarks: MediaPipeの顔ランドマーク
            image_width: 画像幅
            image_height: 画像高さ
            eye_indices: 目のランドマークインデックス
            
        Returns:
            list: [(x, y), ...] 目のランドマーク座標
        """
        landmarks = []
        for idx in eye_indices:
            landmark = face_landmarks.landmark[idx]
            x = landmark.x * image_width
            y = landmark.y * image_height
            landmarks.append((x, y))
        return landmarks
    
    def detect_blink(self, left_ear, right_ear, left_eye_landmarks, right_eye_landmarks):
        """
        瞬きを検出し、完了時に特徴量を抽出
        
        Args:
            left_ear: 左目のEAR
            right_ear: 右目のEAR
            left_eye_landmarks: 左目のランドマーク
            right_eye_landmarks: 右目のランドマーク
            
        Returns:
            dict: 瞬き情報（完了時のみ） または None
        """
        avg_ear = (left_ear + right_ear) / 2.0
        current_time = time.time()
        
        # 状態遷移
        if self.blink_state == self.STATE_OPEN:
            if avg_ear < self.ear_threshold:
                self.blink_state = self.STATE_CLOSING
                self.t1 = current_time
                self.state_start_time = current_time
        
        elif self.blink_state == self.STATE_CLOSING:
            if avg_ear >= self.ear_threshold:
                # 瞬きキャンセル
                self.blink_state = self.STATE_OPEN
            else:
                closing_time = current_time - self.state_start_time
                if closing_time >= 0.01:  # 10ms以上
                    self.blink_state = self.STATE_CLOSED
                    self.t2 = current_time
                    self.state_start_time = current_time
        
        elif self.blink_state == self.STATE_CLOSED:
            if avg_ear >= self.ear_threshold:
                self.blink_state = self.STATE_OPENING
                self.t3 = current_time
                self.state_start_time = current_time
        
        elif self.blink_state == self.STATE_OPENING:
            opening_time = current_time - self.state_start_time
            if opening_time >= 0.01:  # 10ms以上
                # 瞬き完了
                self.blink_state = self.STATE_OPEN
                
                # 特徴量を抽出
                features = self.extract_blink_features(
                    left_eye_landmarks, 
                    right_eye_landmarks
                )
                
                if features is not None:
                    self.total_blinks += 1
                    return features
        
        return None
    
    def extract_blink_features(self, left_eye_landmarks, right_eye_landmarks):
        """
        瞬き完了時に12次元特徴量を抽出
        
        Returns:
            np.array: 12次元特徴量 または None
        """
        try:
            # 時間計算
            closing_time = self.t2 - self.t1
            opening_time = time.time() - self.t3
            total_duration = closing_time + opening_time
            blink_coefficient = opening_time / closing_time if closing_time > 0 else 0
            
            # 有効性チェック
            if not (0.025 <= closing_time <= 1.0):
                return None
            if not (0.05 <= opening_time <= 0.6):
                return None
            if not (0.5 <= blink_coefficient <= 8.0):
                return None
            
            # 2円パラメータを抽出
            c1_left, c2_left = TwoCircleFitter.fit_eyelids(left_eye_landmarks)
            c1_right, c2_right = TwoCircleFitter.fit_eyelids(right_eye_landmarks)
            
            # 両目の平均
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
            
            # 12次元特徴量
            features = np.array([
                closing_time,
                opening_time,
                blink_coefficient,
                self.t1,           # timestamp
                total_duration,
                0.0,               # interval（リアルタイムでは計算困難）
                c1_center_x,
                c1_center_y,
                c1_radius,
                c2_center_x,
                c2_center_y,
                c2_radius
            ], dtype=np.float32)
            
            return features
            
        except Exception as e:
            print(f"⚠️ 特徴量抽出エラー: {e}")
            return None
    
    def predict_drowsiness(self):
        """
        LSTMモデルで眠気を推定
        
        Returns:
            tuple: (prediction, probability) または (None, None)
        """
        if len(self.feature_buffer) < self.sequence_length:
            return None, None
        
        try:
            # シーケンスを作成
            sequence = np.array(list(self.feature_buffer))
            sequence = sequence.reshape(1, self.sequence_length, -1)
            
            # 推論
            proba = self.estimator.predict_proba(sequence)
            prediction = np.argmax(proba[0])
            probability = proba[0][prediction]
            
            return prediction, probability
            
        except Exception as e:
            print(f"⚠️ 推論エラー: {e}")
            return None, None
    
    def draw_info(self, frame, left_ear, right_ear):
        """
        画面に情報を表示
        
        Args:
            frame: 画像フレーム
            left_ear: 左目のEAR
            right_ear: 右目のEAR
        """
        height, width = frame.shape[:2]
        avg_ear = (left_ear + right_ear) / 2.0
        
        # 背景（半透明黒）
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # EAR情報
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 瞬き回数
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # バッファ状態
        buffer_status = f"Buffer: {len(self.feature_buffer)}/{self.sequence_length}"
        cv2.putText(frame, buffer_status, (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 眠気判定
        if self.current_prediction is not None:
            if self.current_prediction == 0:
                label = "Normal"
                color = (0, 255, 0)  # 緑
            else:
                label = "DROWSY!"
                color = (0, 0, 255)  # 赤
                self.drowsy_count += 1
            
            # ラベル表示
            cv2.putText(frame, label, (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
            # 確率表示
            prob_text = f"Confidence: {self.current_probability:.1%}"
            cv2.putText(frame, prob_text, (20, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Collecting data...", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 警告表示
        if self.current_prediction == 1:
            # 画面上部に大きく警告
            warning_text = "!!! DROWSINESS DETECTED !!!"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (width - text_size[0]) // 2
            cv2.putText(frame, warning_text, (text_x, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    def run(self):
        """
        リアルタイム検知を実行
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ カメラを開けません")
            return
        
        print("\n🎥 カメラ起動")
        print("   'q' キーで終了")
        print("=" * 70)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 左右反転（鏡像）
                frame = cv2.flip(frame, 1)
                
                # RGB変換
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 顔検出
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    height, width, _ = frame.shape
                    
                    # 目のランドマーク抽出
                    left_eye = self.extract_eye_landmarks(
                        face_landmarks, width, height, self.LEFT_EYE_INDICES
                    )
                    right_eye = self.extract_eye_landmarks(
                        face_landmarks, width, height, self.RIGHT_EYE_INDICES
                    )
                    
                    # EAR計算
                    left_ear = EARCalculator.calculate(left_eye)
                    right_ear = EARCalculator.calculate(right_eye)
                    
                    # 瞬き検出
                    blink_features = self.detect_blink(
                        left_ear, right_ear, left_eye, right_eye
                    )
                    
                    if blink_features is not None:
                        # バッファに追加
                        self.feature_buffer.append(blink_features)
                        
                        # 眠気推定
                        pred, prob = self.predict_drowsiness()
                        if pred is not None:
                            self.current_prediction = pred
                            self.current_probability = prob
                    
                    # 情報表示
                    self.draw_info(frame, left_ear, right_ear)
                
                # フレーム表示
                cv2.imshow('Drowsiness Detection', frame)
                
                # 'q'キーで終了
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
            if self.total_blinks > 0:
                drowsy_rate = (self.drowsy_count / self.total_blinks) * 100
                print(f"眠気検出率: {drowsy_rate:.1f}%")
            print("=" * 70)


def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(
        description='リアルタイム眠気検知システム',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model-path', type=str, 
                       default='trained_models/drowsiness_lstm_20251115_224046.pth',
                       help='学習済みモデルのパス')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='LSTMのシーケンス長')
    parser.add_argument('--ear-threshold', type=float, default=0.21,
                       help='EAR閾値')
    
    args = parser.parse_args()
    
    # システム起動
    detector = RealtimeDrowsinessDetector(
        model_path=args.model_path,
        sequence_length=args.sequence_length,
        ear_threshold=args.ear_threshold
    )
    
    detector.run()


if __name__ == "__main__":
    main()