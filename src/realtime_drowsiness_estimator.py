"""
リアルタイム眠気推定システム
Real-time Drowsiness Estimation System

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
    from src.blink_detector import BlinkDetector
    from src.blink_feature_extractor import BlinkFeatureExtractor
    from src.lstm_drowsiness_model import DrowsinessEstimator
except ImportError as e:
    print(f"❌ モジュールのインポートエラー: {e}")
    print("   必要なファイル: src/blink_detector.py, src/blink_feature_extractor.py, src/lstm_drowsiness_model.py")
    print("   プロジェクトルートから実行してください: python -m src.realtime_drowsiness_estimator")
    print("   必要なファイル: blink_detector.py, blink_feature_extractor.py, lstm_drowsiness_model.py")
    sys.exit(1)


class RealtimeDrowsinessEstimator:
    """
    リアルタイム眠気推定システム
    
    機能:
    - カメラからリアルタイムで瞬き検出
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
        print("🚀 リアルタイム眠気推定システム初期化")
        print("=" * 70)
        
        # 瞬き検出器
        print("\n📹 瞬き検出器初期化...")
        self.blink_detector = BlinkDetector()
        
        # 特徴量抽出器
        print("🔧 特徴量抽出器初期化...")
        self.feature_extractor = BlinkFeatureExtractor(sequence_length=sequence_length)
        
        # 正規化パラメータ読み込み
        if normalization_params_path and os.path.exists(normalization_params_path):
            self.feature_extractor.load_normalization_params(normalization_params_path)
        
        # モデル読み込み
        print("🧠 LSTMモデル読み込み...")
        self.estimator = DrowsinessEstimator()
        self.estimator.load_model(model_path)
        
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
        
        # アラート管理
        self.alert_active = False
        self.alert_start_time = None
        self.alert_count = 0
        self.consecutive_drowsy_count = 0
        self.consecutive_drowsy_threshold = 3  # 連続3回で警告
        
        # 統計情報
        self.stats = {
            'start_time': None,
            'total_frames': 0,
            'total_blinks': 0,
            'total_predictions': 0,
            'normal_predictions': 0,
            'drowsy_predictions': 0,
            'alert_count': 0,
            'session_duration': 0
        }
        
        # UI設定
        self.window_name = "眠気推定システム"
        self.show_info = True
        self.show_graph = True
        
        # ログ設定
        self.log_dir = "drowsiness_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"session_{self.session_id}.json")
        
        print("✅ 初期化完了")
        print("=" * 70)
    
    def initialize_camera(self, camera_id: int = 0) -> bool:
        """
        カメラを初期化
        
        Args:
            camera_id (int): カメラID
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            self.camera = cv2.VideoCapture(camera_id)
            
            if not self.camera.isOpened():
                print("❌ カメラを開けませんでした")
                return False
            
            # カメラ設定
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            print(f"\n📹 カメラ初期化完了")
            print(f"   解像度: {self.camera_width}x{self.camera_height}")
            print(f"   FPS: {self.fps}")
            
            # キャリブレーション
            print(f"\n🎯 個人キャリブレーション開始（5秒間）")
            print("   リラックスして自然に瞬きしてください...")
            
            self.blink_detector.start_calibration()
            
            calib_start = time.time()
            while time.time() - calib_start < 5.0:
                ret, frame = self.camera.read()
                if ret:
                    # 顔検出（OpenCV Haar Cascade使用）
                    face_rect = self.blink_detector.detect_face(frame)
                    
                    if face_rect is not None:
                        # 瞬き検出（キャリブレーション用）
                        blink_detected, ear, blink_state = self.blink_detector.detect_blink(frame, face_rect)
                        
                        # キャリブレーション中の映像を表示
                        if self.show_visualization:
                            display_frame = frame.copy()
                            self.blink_detector.draw_debug_info(display_frame, face_rect)
                            cv2.imshow("キャリブレーション", display_frame)
                            cv2.waitKey(1)
            
            print("✅ キャリブレーション完了")
            
            return True
            
        except Exception as e:
            print(f"❌ カメラ初期化エラー: {e}")
            return False
    
    def process_frame(self, frame) -> Tuple[np.ndarray, Dict]:
        """
        1フレームを処理
        
        Args:
            frame: 入力フレーム
            
        Returns:
            tuple: (処理済みフレーム, 推定結果)
        """
        self.stats['total_frames'] += 1
        
        # 顔検出（OpenCV Haar Cascade使用）
        face_rect = self.blink_detector.detect_face(frame)
        
        ear = None
        blink_detected = False
        blink_state = None
        
        if face_rect is not None:
            # 瞬き検出
            blink_detected, ear, blink_state = self.blink_detector.detect_blink(frame, face_rect)
        
        result = {
            'ear': ear,
            'blink_detected': blink_detected,
            'state': self.current_state,
            'probability': self.current_probability,
            'alert': self.alert_active
        }
        
        # 瞬きが検出された場合
        if blink_detected:
            self.stats['total_blinks'] += 1
            
            # 瞬きデータ取得
            blink_data = self._get_blink_data()
            
            if blink_data is not None:
                # 特徴量抽出
                features = self.feature_extractor.extract_features(blink_data)
                
                if features is not None:
                    # シーケンスデータ取得
                    sequence = self.feature_extractor.get_sequence(normalize=True)
                    
                    if sequence is not None:
                        # 推定実行
                        prediction_result = self._predict_drowsiness(sequence)
                        result.update(prediction_result)
                        
                        # 状態更新
                        self._update_state(prediction_result)
                        
                        # アラートチェック
                        self._check_alert()
        
        # 可視化
        frame = self._draw_ui(frame, result)
        
        return frame, result
    
    def _get_blink_data(self) -> Optional[Dict]:
        """
        最新の瞬きデータを取得
        
        Returns:
            Dict: 瞬きデータ
        """
        if len(self.blink_detector.blink_details) == 0:
            return None
        
        latest_blink = self.blink_detector.blink_details[-1]
        
        required_keys = ['t1', 't2', 't3', 'ear_min']
        if not all(key in latest_blink for key in required_keys):
            return None
        
        return {
            't1': latest_blink['t1'],
            't2': latest_blink['t2'],
            't3': latest_blink['t3'],
            'ear_min': latest_blink['ear_min']
        }
    
    def _predict_drowsiness(self, sequence: np.ndarray) -> Dict:
        """
        眠気を推定
        
        Args:
            sequence: 入力シーケンス (10, 6)
            
        Returns:
            Dict: 推定結果
        """
        # バッチ次元追加
        sequence_batch = sequence[np.newaxis, ...]
        
        # 推定
        pred_class = self.estimator.predict(sequence_batch)[0]
        pred_proba = self.estimator.predict_proba(sequence_batch)[0]
        
        result = {
            'class': int(pred_class),
            'state': int(pred_class),
            'normal_probability': float(pred_proba[0]),
            'drowsy_probability': float(pred_proba[1]),
            'probability': float(pred_proba[1]),  # 眠気確率
            'confidence': float(max(pred_proba))
        }
        
        self.last_prediction_time = time.time()
        self.stats['total_predictions'] += 1
        
        if pred_class == self.STATE_NORMAL:
            self.stats['normal_predictions'] += 1
        else:
            self.stats['drowsy_predictions'] += 1
        
        return result
    
    def _update_state(self, prediction_result: Dict):
        """
        システム状態を更新
        
        Args:
            prediction_result: 推定結果
        """
        self.current_state = prediction_result['state']
        self.current_probability = prediction_result['probability']
        
        # 履歴に追加
        self.prediction_history.append(self.current_state)
        self.drowsy_probability_history.append(self.current_probability)
        
        # 連続眠気カウント
        if self.current_state == self.STATE_DROWSY:
            self.consecutive_drowsy_count += 1
        else:
            self.consecutive_drowsy_count = 0
    
    def _check_alert(self):
        """
        アラート条件をチェック
        """
        # アラート条件
        should_alert = (
            self.current_state == self.STATE_DROWSY and
            self.current_probability >= self.alert_threshold and
            self.consecutive_drowsy_count >= self.consecutive_drowsy_threshold
        )
        
        if should_alert and not self.alert_active:
            # アラート開始
            self.alert_active = True
            self.alert_start_time = time.time()
            self.alert_count += 1
            self.stats['alert_count'] += 1
            print(f"\n⚠️ 【警告】眠気検出！ (確率: {self.current_probability:.1%})")
        
        elif not should_alert and self.alert_active:
            # アラート解除
            self.alert_active = False
            alert_duration = time.time() - self.alert_start_time
            print(f"✅ アラート解除 (継続時間: {alert_duration:.1f}秒)")
    
    def _draw_ui(self, frame, result: Dict) -> np.ndarray:
        """
        UIを描画
        
        Args:
            frame: 元フレーム
            result: 推定結果
            
        Returns:
            処理済みフレーム
        """
        h, w = frame.shape[:2]
        
        # アラート表示（全画面フラッシュ）
        if self.alert_active:
            # 赤いオーバーレイ
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            alpha = 0.3 + 0.2 * np.sin(time.time() * 10)  # 点滅効果
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # 情報パネル背景
        panel_height = 220
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # 状態表示
        state_text = "不明"
        state_color = (128, 128, 128)
        
        if result['state'] == self.STATE_NORMAL:
            state_text = "正常"
            state_color = (0, 255, 0)
        elif result['state'] == self.STATE_DROWSY:
            state_text = "眠気"
            state_color = (0, 165, 255)
        
        cv2.putText(frame, f"Status: {state_text}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3)
        
        # 確率表示
        if result['probability'] > 0:
            prob_text = f"Drowsy Prob: {result['probability']:.1%}"
            cv2.putText(frame, prob_text, (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # EAR値表示
        if result['ear'] is not None:
            ear_text = f"EAR: {result['ear']:.3f}"
            cv2.putText(frame, ear_text, (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 瞬き表示
        if result['blink_detected']:
            cv2.putText(frame, "BLINK!", (20, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 統計表示
        blink_text = f"Blinks: {self.stats['total_blinks']}"
        cv2.putText(frame, blink_text, (20, 175),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        pred_text = f"Predictions: {self.stats['total_predictions']}"
        cv2.putText(frame, pred_text, (20, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # アラート表示
        if self.alert_active:
            alert_text = "!!! DROWSINESS ALERT !!!"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 50
            
            # 背景
            cv2.rectangle(frame, (text_x - 10, text_y - 40),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 255), -1)
            
            # テキスト
            cv2.putText(frame, alert_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        
        # グラフ表示
        if self.show_graph and len(self.drowsy_probability_history) > 1:
            frame = self._draw_probability_graph(frame)
        
        return frame
    
    def _draw_probability_graph(self, frame) -> np.ndarray:
        """
        眠気確率のグラフを描画
        
        Args:
            frame: 元フレーム
            
        Returns:
            グラフ付きフレーム
        """
        h, w = frame.shape[:2]
        
        # グラフ領域
        graph_x = w - 310
        graph_y = h - 160
        graph_w = 300
        graph_h = 150
        
        # 背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (graph_x, graph_y),
                     (graph_x + graph_w, graph_y + graph_h),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # 枠
        cv2.rectangle(frame, (graph_x, graph_y),
                     (graph_x + graph_w, graph_y + graph_h),
                     (255, 255, 255), 2)
        
        # 閾値線
        threshold_y = graph_y + graph_h - int(self.alert_threshold * graph_h)
        cv2.line(frame, (graph_x, threshold_y),
                (graph_x + graph_w, threshold_y),
                (0, 165, 255), 1)
        
        # データプロット
        history = list(self.drowsy_probability_history)
        if len(history) > 1:
            points = []
            for i, prob in enumerate(history[-graph_w:]):
                x = graph_x + i
                y = graph_y + graph_h - int(prob * graph_h)
                points.append((x, y))
            
            # 線描画
            for i in range(len(points) - 1):
                color = (0, 255, 0) if history[-(graph_w - i)] < self.alert_threshold else (0, 0, 255)
                cv2.line(frame, points[i], points[i + 1], color, 2)
        
        # ラベル
        cv2.putText(frame, "Drowsy Probability", (graph_x + 5, graph_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """
        メインループを実行
        """
        if self.camera is None or not self.camera.isOpened():
            print("❌ カメラが初期化されていません")
            return
        
        print("\n" + "=" * 70)
        print("🎬 リアルタイム推定開始")
        print("=" * 70)
        print("\n操作方法:")
        print("  [SPACE] - 統計情報表示")
        print("  [R]     - 統計リセット")
        print("  [G]     - グラフ表示切替")
        print("  [S]     - セッション保存")
        print("  [ESC]   - 終了")
        print("=" * 70)
        
        self.stats['start_time'] = time.time()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ フレーム取得失敗")
                    break
                
                # フレーム処理
                processed_frame, result = self.process_frame(frame)
                
                # 表示
                cv2.imshow(self.window_name, processed_frame)
                
                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\n👋 終了します")
                    break
                
                elif key == ord(' '):  # SPACE
                    self.print_statistics()
                
                elif key == ord('r') or key == ord('R'):
                    self.reset_statistics()
                
                elif key == ord('g') or key == ord('G'):
                    self.show_graph = not self.show_graph
                    print(f"📊 グラフ表示: {'ON' if self.show_graph else 'OFF'}")
                
                elif key == ord('s') or key == ord('S'):
                    self.save_session()
        
        finally:
            self.cleanup()
    
    def print_statistics(self):
        """
        統計情報を表示
        """
        if self.stats['start_time'] is not None:
            self.stats['session_duration'] = time.time() - self.stats['start_time']
        
        print("\n" + "=" * 70)
        print("📊 セッション統計")
        print("=" * 70)
        print(f"セッション時間: {self.stats['session_duration']:.1f}秒")
        print(f"総フレーム数: {self.stats['total_frames']}")
        print(f"総瞬き数: {self.stats['total_blinks']}")
        print(f"総推定回数: {self.stats['total_predictions']}")
        
        if self.stats['total_predictions'] > 0:
            normal_rate = 100.0 * self.stats['normal_predictions'] / self.stats['total_predictions']
            drowsy_rate = 100.0 * self.stats['drowsy_predictions'] / self.stats['total_predictions']
            print(f"  正常: {self.stats['normal_predictions']} ({normal_rate:.1f}%)")
            print(f"  眠気: {self.stats['drowsy_predictions']} ({drowsy_rate:.1f}%)")
        
        print(f"アラート回数: {self.stats['alert_count']}")
        print("=" * 70)
    
    def reset_statistics(self):
        """
        統計情報をリセット
        """
        self.stats = {
            'start_time': time.time(),
            'total_frames': 0,
            'total_blinks': 0,
            'total_predictions': 0,
            'normal_predictions': 0,
            'drowsy_predictions': 0,
            'alert_count': 0,
            'session_duration': 0
        }
        self.prediction_history.clear()
        self.drowsy_probability_history.clear()
        print("🔄 統計情報をリセットしました")
    
    def save_session(self):
        """
        セッション情報を保存
        """
        if self.stats['start_time'] is not None:
            self.stats['session_duration'] = time.time() - self.stats['start_time']
        
        session_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'statistics': self.stats,
            'prediction_history': list(self.prediction_history),
            'drowsy_probability_history': list(self.drowsy_probability_history)
        }
        
        try:
            with open(self.log_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            print(f"💾 セッション保存: {self.log_file}")
        except Exception as e:
            print(f"❌ セッション保存エラー: {e}")
    
    def cleanup(self):
        """
        リソースを解放
        """
        print("\n🧹 クリーンアップ中...")
        
        # 最終統計表示
        self.print_statistics()
        
        # セッション保存
        self.save_session()
        
        # カメラ解放
        if self.camera is not None:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        print("✅ クリーンアップ完了")


def parse_args():
    """
    コマンドライン引数をパース
    """
    parser = argparse.ArgumentParser(
        description='リアルタイム眠気推定システム',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='訓練済みモデルのパス (.pth)')
    parser.add_argument('--norm-params', type=str, default=None,
                       help='正規化パラメータのパス (.json)')
    parser.add_argument('--camera', type=int, default=0,
                       help='カメラID')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='シーケンス長')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='アラート閾値 (0.0-1.0)')
    
    return parser.parse_args()


def main():
    """
    メイン関数
    """
    print("=" * 70)
    print("🚀 リアルタイム眠気推定システム")
    print("=" * 70)
    
    # 引数パース
    args = parse_args()
    
    # モデルファイル確認
    if not os.path.exists(args.model):
        print(f"❌ モデルファイルが見つかりません: {args.model}")
        print("\n使用方法:")
        print("  python realtime_drowsiness_estimator.py --model <model_path>")
        print("\n例:")
        print("  python realtime_drowsiness_estimator.py \\")
        print("    --model trained_models/drowsiness_lstm_20240101_120000.pth \\")
        print("    --norm-params drowsiness_training_data/normalization_params.json")
        sys.exit(1)
    
    # 正規化パラメータのデフォルトパス
    if args.norm_params is None:
        default_norm_path = "drowsiness_training_data/normalization_params.json"
        if os.path.exists(default_norm_path):
            args.norm_params = default_norm_path
            print(f"📊 正規化パラメータを自動検出: {default_norm_path}")
    
    # システム作成
    estimator = RealtimeDrowsinessEstimator(
        model_path=args.model,
        normalization_params_path=args.norm_params,
        sequence_length=args.sequence_length,
        alert_threshold=args.threshold
    )
    
    # カメラ初期化
    if not estimator.initialize_camera(args.camera):
        print("❌ カメラ初期化に失敗しました")
        sys.exit(1)
    
    # 実行
    estimator.run()


if __name__ == "__main__":
    main()