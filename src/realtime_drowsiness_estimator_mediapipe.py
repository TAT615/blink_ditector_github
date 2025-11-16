"""
リアルタイム眠気推定システム（MediaPipe版）- 修正版
Real-time Drowsiness Estimation System with MediaPipe - FIXED

訓練済みLSTMモデルを使用してリアルタイムで眠気を推定します。
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

# 自作モジュールのインポート
try:
    from src.blink_feature_extractor import BlinkFeatureExtractor
    from src.lstm_drowsiness_model import DrowsinessEstimator
except ImportError:
    try:
        from blink_feature_extractor import BlinkFeatureExtractor
        from lstm_drowsiness_model import DrowsinessEstimator
    except ImportError as e:
        print(f"❌ モジュールのインポートエラー: {e}")
        print("   必要なファイル: blink_feature_extractor.py, lstm_drowsiness_model.py")
        sys.exit(1)

# MediaPipe版瞬き検出器をインポート
try:
    from blink_detector_mediapipe import BlinkDetectorMediaPipe
except ImportError:
    try:
        from src.blink_detector_mediapipe import BlinkDetectorMediaPipe
    except ImportError as e:
        print(f"❌ MediaPipe版瞬き検出器のインポートエラー: {e}")
        print("   blink_detector_mediapipe.py が必要です")
        sys.exit(1)


class RealtimeDrowsinessEstimatorMediaPipe:
    """
    リアルタイム眠気推定システム（MediaPipe版）
    
    機能:
    - MediaPipe Face Meshによる高精度瞬き検出
    - 特徴量抽出とシーケンス生成
    - LSTM推論による眠気推定
    - アラート機能
    - 統計記録
    """
    
    # 状態定義
    STATE_NORMAL = 0
    STATE_DROWSY = 1
    STATE_UNKNOWN = -1
    
    def __init__(self, model_path: str, normalization_params_path: Optional[str] = None,
                 sequence_length: int = 10, alert_threshold: float = 0.7):
        """
        初期化
        
        Args:
            model_path (str): 訓練済みモデルのパス
            normalization_params_path (str): 正規化パラメータのパス
            sequence_length (int): シーケンス長
            alert_threshold (float): アラートを発する眠気確率の閾値
        """
        self.model_path = model_path
        self.normalization_params_path = normalization_params_path
        self.sequence_length = sequence_length
        self.alert_threshold = alert_threshold
        
        # モジュールの初期化
        print("=" * 70)
        print("🚀 リアルタイム眠気推定システム初期化 (MediaPipe版)")
        print("=" * 70)
        
        # MediaPipe版瞬き検出器
        print("\n📹 MediaPipe Face Mesh初期化...")
        self.blink_detector = BlinkDetectorMediaPipe(
            buffer_size=300,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("   ✅ 478ランドマーク高精度顔検出")
        
        # 特徴量抽出器
        print("🔧 特徴量抽出器初期化...")
        self.feature_extractor = BlinkFeatureExtractor(sequence_length=sequence_length)
        
        # 正規化パラメータ読み込み
        if normalization_params_path and os.path.exists(normalization_params_path):
            self.feature_extractor.load_normalization_params(normalization_params_path)
            print(f"   ✅ 正規化パラメータ読み込み: {normalization_params_path}")
        else:
            print("   ⚠️ 正規化パラメータなし（訓練データで正規化してください）")
        
        # モデル読み込み
        print("🧠 LSTMモデル読み込み...")
        self.estimator = DrowsinessEstimator()
        self.estimator.load_model(model_path)
        print(f"   ✅ モデル読み込み成功: {model_path}")
        
        # カメラ設定
        self.camera = None
        self.camera_width = 640
        self.camera_height = 480
        self.fps = 30
        
        # 推定結果の履歴
        self.prediction_history = deque(maxlen=100)
        self.drowsy_probability_history = deque(maxlen=100)
        
        # 現在の状態
        self.current_state = self.STATE_UNKNOWN
        self.current_probability = 0.0
        self.last_prediction_time = None
        
        # アラート設定
        self.alert_active = False
        self.alert_start_time = None
        self.consecutive_drowsy_count = 0
        self.alert_cooldown = 5.0  # アラート後のクールダウン時間（秒）
        
        # 統計
        self.stats = {
            'total_predictions': 0,
            'drowsy_predictions': 0,
            'normal_predictions': 0,
            'total_alerts': 0,
            'session_start_time': time.time()
        }
        
        # UI設定
        self.window_name = "リアルタイム眠気推定 (MediaPipe版)"
        
        print("\n" + "=" * 70)
        print("✅ 初期化完了")
        print("=" * 70)
    
    def initialize_camera(self, camera_id=0):
        """
        カメラを初期化
        
        Args:
            camera_id (int): カメラID
            
        Returns:
            bool: 成功したかどうか
        """
        self.camera = cv2.VideoCapture(camera_id)
        
        if not self.camera.isOpened():
            print(f"❌ カメラ {camera_id} を開けませんでした")
            return False
        
        # カメラ設定
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        
        print(f"✅ カメラ初期化成功 (ID: {camera_id})")
        print(f"   解像度: {self.camera_width}x{self.camera_height}")
        print(f"   FPS: {self.fps}")
        
        return True
    
    def start_calibration(self):
        """キャリブレーションを開始"""
        print("\n" + "=" * 70)
        print("🎯 キャリブレーション開始")
        print("=" * 70)
        print("次の5秒間、リラックスして自然に瞬きしてください")
        print()
        
        self.blink_detector.start_calibration()
    
    def process_frame(self, frame):
        """
        フレームを処理
        
        Args:
            frame: カメラフレーム
            
        Returns:
            tuple: (処理済みフレーム, 推定結果)
        """
        # フレームを左右反転
        frame = cv2.flip(frame, 1)
        
        # MediaPipeで瞬き検出
        blink_info = self.blink_detector.detect_blink(frame)
        
        # ランドマークを取得して描画
        landmarks = self.blink_detector.detect_face_and_landmarks(frame)
        if landmarks is not None:
            frame = self.blink_detector.draw_landmarks(frame, landmarks)
        
        # 推定結果
        prediction_result = None
        
        # 瞬きが検出された場合
        if blink_info is not None:
            # ====== 修正箇所: extract_featuresメソッドを使用 ======
            # 瞬きデータを適切な形式に変換
            blink_data = {
                't1': blink_info.get('timestamp', 0) - blink_info.get('total_duration', 0),
                't2': blink_info.get('timestamp', 0) - blink_info.get('opening_time', 0),
                't3': blink_info.get('timestamp', 0),
                'ear_min': blink_info.get('min_ear', 0)
            }
            
            # extract_featuresメソッドを使用
            features = self.feature_extractor.extract_features(blink_data)
            
            if features is not None:
                # シーケンスが溜まったら推定実行
                sequence = self.feature_extractor.get_sequence(normalize=True)
                
                if sequence is not None:
                    # LSTM推論
                    pred_class = self.estimator.predict(sequence[np.newaxis, ...])[0]
                    pred_proba = self.estimator.predict_proba(sequence[np.newaxis, ...])[0]
                    
                    # 結果を記録
                    self.prediction_history.append(pred_class)
                    self.drowsy_probability_history.append(pred_proba[1])
                    
                    # 統計更新
                    self.stats['total_predictions'] += 1
                    if pred_class == self.STATE_DROWSY:
                        self.stats['drowsy_predictions'] += 1
                        self.consecutive_drowsy_count += 1
                    else:
                        self.stats['normal_predictions'] += 1
                        self.consecutive_drowsy_count = 0
                    
                    # 状態更新
                    self.current_state = pred_class
                    self.current_probability = pred_proba[1]
                    self.last_prediction_time = time.time()
                    
                    # アラートチェック
                    self._check_alert()
                    
                    prediction_result = {
                        'class': pred_class,
                        'probability': pred_proba[1],
                        'state': 'DROWSY' if pred_class == self.STATE_DROWSY else 'NORMAL'
                    }
        
        return frame, prediction_result
    
    def _check_alert(self):
        """アラートをチェック"""
        current_time = time.time()
        
        # 眠気確率が閾値を超え、かつ連続検出の場合
        if (self.current_probability >= self.alert_threshold and 
            self.consecutive_drowsy_count >= 3):
            
            # クールダウン中でなければアラート発動
            if (self.alert_start_time is None or 
                current_time - self.alert_start_time >= self.alert_cooldown):
                
                self.alert_active = True
                self.alert_start_time = current_time
                self.stats['total_alerts'] += 1
                
                print(f"\n⚠️ 【アラート】眠気を検出しました！ (確率: {self.current_probability:.1%})")
                print(f"   休憩を取ることをお勧めします\n")
        else:
            self.alert_active = False
    
    def draw_ui(self, frame):
        """
        UIを描画
        
        Args:
            frame: フレーム
            
        Returns:
            frame: UI描画済みフレーム
        """
        h, w = frame.shape[:2]
        
        # 半透明の背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        y_offset = 30
        line_height = 30
        
        # タイトル
        cv2.putText(frame, "Drowsiness Estimation (MediaPipe)", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height
        
        # 現在の状態
        if self.current_state == self.STATE_DROWSY:
            state_text = "DROWSY"
            state_color = (0, 0, 255)  # 赤
        elif self.current_state == self.STATE_NORMAL:
            state_text = "NORMAL"
            state_color = (0, 255, 0)  # 緑
        else:
            state_text = "UNKNOWN"
            state_color = (128, 128, 128)  # グレー
        
        cv2.putText(frame, f"State: {state_text}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        y_offset += line_height
        
        # 眠気確率
        prob_text = f"Drowsy Prob: {self.current_probability:.1%}"
        prob_color = (0, 255, 0) if self.current_probability < 0.5 else (0, 0, 255)
        cv2.putText(frame, prob_text, 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, prob_color, 2)
        y_offset += line_height
        
        # 確率バー
        bar_width = 300
        bar_height = 20
        bar_x = 10
        bar_y = y_offset
        
        # 背景
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # 確率バー
        prob_bar_width = int(bar_width * self.current_probability)
        bar_color = (0, 255, 0) if self.current_probability < 0.5 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + prob_bar_width, bar_y + bar_height), 
                     bar_color, -1)
        
        # 閾値ライン
        threshold_x = bar_x + int(bar_width * self.alert_threshold)
        cv2.line(frame, (threshold_x, bar_y), (threshold_x, bar_y + bar_height), 
                (255, 255, 255), 2)
        
        y_offset += bar_height + 15
        
        # 検出器の統計
        detector_stats = self.blink_detector.get_statistics()
        
        # EAR値
        ear = detector_stats['current_ear']
        ear_color = (0, 255, 0)
        if self.blink_detector.ear_closed_threshold and ear <= self.blink_detector.ear_closed_threshold:
            ear_color = (0, 0, 255)
        
        cv2.putText(frame, f"EAR: {ear:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
        y_offset += line_height
        
        # 瞬き統計
        cv2.putText(frame, f"Blinks: {detector_stats['total_blinks']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        # 推定回数
        cv2.putText(frame, f"Predictions: {self.stats['total_predictions']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        # アラート回数
        cv2.putText(frame, f"Alerts: {self.stats['total_alerts']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += line_height
        
        # キャリブレーション状態
        if self.blink_detector.calibration_active:
            elapsed = time.time() - self.blink_detector.calibration_start_time
            remaining = self.blink_detector.calibration_duration - elapsed
            
            cv2.putText(frame, f"Calibrating: {remaining:.1f}s", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            calib_text = "Calibrated: YES" if detector_stats['calibrated'] else "NOT Calibrated (Press C)"
            calib_color = (0, 255, 0) if detector_stats['calibrated'] else (0, 0, 255)
            cv2.putText(frame, calib_text, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, calib_color, 2)
        
        # アラート表示
        if self.alert_active:
            alert_text = "⚠️ DROWSINESS ALERT!"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            alert_x = (w - text_size[0]) // 2
            alert_y = h - 100
            
            # 点滅効果
            if int(time.time() * 3) % 2 == 0:
                cv2.putText(frame, alert_text, 
                           (alert_x, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        return frame
    
    def run(self):
        """
        システムを実行
        """
        # カメラ初期化
        if not self.initialize_camera():
            return
        
        print("\n" + "=" * 70)
        print("🚀 リアルタイム眠気推定システム起動")
        print("=" * 70)
        print("操作方法:")
        print("  [C] - キャリブレーション（最初に実行推奨）")
        print("  [R] - 統計リセット")
        print("  [ESC] - 終了")
        print(f"アラート閾値: {self.alert_threshold:.0%}")
        print("=" * 70)
        print("👉 まず[C]キーでキャリブレーションを実行してください")
        
        # FPS計測
        fps = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ フレーム取得失敗")
                    break
                
                # FPS計算
                fps_frame_count += 1
                if fps_frame_count >= 30:
                    fps_end_time = time.time()
                    fps = fps_frame_count / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    fps_frame_count = 0
                
                # フレーム処理
                frame, prediction = self.process_frame(frame)
                
                # UI描画
                frame = self.draw_ui(frame)
                
                # FPS表示
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                           (frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 表示
                cv2.imshow(self.window_name, frame)
                
                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('c') or key == ord('C'):
                    self.start_calibration()
                elif key == ord('r') or key == ord('R'):
                    print("\n🔄 統計をリセットしました")
                    self.stats = {
                        'total_predictions': 0,
                        'drowsy_predictions': 0,
                        'normal_predictions': 0,
                        'total_alerts': 0,
                        'session_start_time': time.time()
                    }
        
        finally:
            # 終了処理
            self.cleanup()
    
    def cleanup(self):
        """リソースを解放"""
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        # 最終統計
        session_duration = time.time() - self.stats['session_start_time']
        
        print("\n" + "=" * 70)
        print("📊 セッション統計")
        print("=" * 70)
        print(f"セッション時間: {session_duration/60:.1f}分")
        print(f"総推定回数: {self.stats['total_predictions']}")
        print(f"  - 正常: {self.stats['normal_predictions']}")
        print(f"  - 眠気: {self.stats['drowsy_predictions']}")
        print(f"総アラート回数: {self.stats['total_alerts']}")
        
        if self.stats['total_predictions'] > 0:
            drowsy_rate = self.stats['drowsy_predictions'] / self.stats['total_predictions']
            print(f"眠気検出率: {drowsy_rate:.1%}")
        
        print("=" * 70)
        print()
        print("リアルタイム眠気推定システムを終了しました")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="リアルタイム眠気推定システム (MediaPipe版)")
    parser.add_argument('--model', type=str, required=True,
                       help='訓練済みモデルのパス')
    parser.add_argument('--norm-params', type=str, default=None,
                       help='正規化パラメータのパス')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='シーケンス長（デフォルト: 10）')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='アラート閾値（デフォルト: 0.7）')
    parser.add_argument('--camera', type=int, default=0,
                       help='カメラID（デフォルト: 0）')
    
    args = parser.parse_args()
    
    # システム初期化
    estimator = RealtimeDrowsinessEstimatorMediaPipe(
        model_path=args.model,
        normalization_params_path=args.norm_params,
        sequence_length=args.sequence_length,
        alert_threshold=args.threshold
    )
    
    # 実行
    estimator.run()


if __name__ == "__main__":
    main()