"""
眠気推定用データ収集システム（MediaPipe版）
Drowsiness Data Collection System with MediaPipe

瞬き特徴量を収集し、正常/眠気状態のラベル付きデータセットを構築します。
"""

import os
import sys
import time
import cv2
import numpy as np
import json
import csv
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple

# MediaPipe版瞬き検出器のインポート
try:
    # プロジェクトのsrcディレクトリから既存モジュールをインポート
    from src.blink_feature_extractor import BlinkFeatureExtractor
except ImportError:
    try:
        # カレントディレクトリからインポート（スタンドアロン実行時）
        from blink_feature_extractor import BlinkFeatureExtractor
    except ImportError as e:
        print(f"⚠️ モジュールのインポートエラー: {e}")
        print("   blink_feature_extractor.py が必要です")

# MediaPipe版瞬き検出器をインポート
try:
    from src.blink_detector_mediapipe import BlinkDetectorMediaPipe
except ImportError:
    try:
        from blink_detector_mediapipe import BlinkDetectorMediaPipe
    except ImportError as e:
        print(f"⚠️ MediaPipe版瞬き検出器のインポートエラー: {e}")
        print("   blink_detector_mediapipe.py が必要です")
        sys.exit(1)


class DrowsinessDataCollectorMediaPipe:
    """
    眠気推定用データ収集システム（MediaPipe版）
    
    機能:
    - MediaPipe Face Meshによる高精度瞬き検出
    - 特徴量抽出
    - 正常/眠気状態のラベリング
    - データの保存（CSV/JSON）
    - セッション管理
    """
    
    # ラベル定義
    LABEL_NORMAL = 0  # 正常状態
    LABEL_DROWSY = 1  # 眠気状態
    
    def __init__(self, data_dir="drowsiness_training_data", sequence_length=10):
        """
        初期化
        
        Args:
            data_dir (str): データ保存ディレクトリ
            sequence_length (int): シーケンス長（LSTM入力用）
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        
        # ディレクトリ作成
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'sessions'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'sequences'), exist_ok=True)
        
        # MediaPipe版瞬き検出器
        self.blink_detector = BlinkDetectorMediaPipe(
            buffer_size=300,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 特徴量抽出器
        self.feature_extractor = BlinkFeatureExtractor(sequence_length=sequence_length)
        
        # 現在のセッション
        self.current_session = None
        self.session_data = []
        self.session_label = self.LABEL_NORMAL  # デフォルトは正常
        self.collecting = False
        
        # 統計情報
        self.stats = {
            'total_sessions': 0,
            'normal_sessions': 0,
            'drowsy_sessions': 0,
            'total_blinks': 0,
            'total_sequences': 0
        }
        
        # カメラ設定
        self.camera = None
        self.camera_width = 640
        self.camera_height = 480
        self.fps = 30
        
        # UI設定
        self.show_visualization = True
        self.window_name = "眠気データ収集システム (MediaPipe版)"
        
        print("=" * 70)
        print("🎯 眠気推定用データ収集システム初期化完了 (MediaPipe版)")
        print("=" * 70)
        print(f"📁 データ保存先: {self.data_dir}")
        print(f"📊 シーケンス長: {self.sequence_length}")
        print(f"🔬 顔検出: MediaPipe Face Mesh (478ランドマーク)")
    
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
        print("顔をカメラに向けて、なるべく動かさないでください")
        print()
        
        self.blink_detector.start_calibration()
    
    def start_session(self, label):
        """
        新しいセッションを開始
        
        Args:
            label (int): LABEL_NORMAL または LABEL_DROWSY
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_name = "normal" if label == self.LABEL_NORMAL else "drowsy"
        
        self.current_session = {
            'session_id': session_id,
            'label': label,
            'label_name': label_name,
            'start_time': time.time(),
            'blink_count': 0
        }
        
        self.session_data = []
        self.session_label = label
        self.collecting = True
        
        print("\n" + "=" * 70)
        print(f"📝 新しいセッション開始: {session_id}")
        print(f"   ラベル: {label_name.upper()}")
        print("=" * 70)
        print("自然に瞬きしてください。[SPACE]で保存、[ESC]でキャンセル")
        print()
    
    def process_frame(self, frame):
        """
        フレームを処理して瞬きを検出
        
        Args:
            frame: カメラフレーム
            
        Returns:
            tuple: (処理済みフレーム, 瞬き情報)
        """
        # フレームを左右反転（鏡像表示）
        frame = cv2.flip(frame, 1)
        
        # MediaPipeで瞬き検出
        blink_info = self.blink_detector.detect_blink(frame)
        
        # 顔とランドマークを取得
        landmarks = self.blink_detector.detect_face_and_landmarks(frame)
        
        # ランドマークを描画
        if landmarks is not None:
            frame = self.blink_detector.draw_landmarks(frame, landmarks)
        
        # 瞬きが検出された場合、データを記録
        if blink_info is not None and self.collecting:
            self.session_data.append(blink_info)
            self.current_session['blink_count'] += 1
            
            print(f"   瞬き #{self.current_session['blink_count']}: "
                  f"係数={blink_info['blink_coefficient']:.2f}, "
                  f"Tc={blink_info['closing_time']*1000:.0f}ms, "
                  f"To={blink_info['opening_time']*1000:.0f}ms")
        
        return frame, blink_info
    
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
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        y_offset = 30
        line_height = 30
        
        # タイトル
        cv2.putText(frame, "Drowsiness Data Collector (MediaPipe)", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height
        
        # セッション情報
        if self.current_session:
            label_color = (0, 255, 0) if self.session_label == self.LABEL_NORMAL else (0, 0, 255)
            label_text = "NORMAL" if self.session_label == self.LABEL_NORMAL else "DROWSY"
            
            cv2.putText(frame, f"Session: {self.current_session['session_id']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
            
            cv2.putText(frame, f"Label: {label_text}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
            y_offset += line_height
            
            cv2.putText(frame, f"Blinks: {self.current_session['blink_count']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
        
        # 検出器の統計
        stats = self.blink_detector.get_statistics()
        
        # EAR値（色分け）
        ear = stats['current_ear']
        ear_color = (0, 255, 0)
        if self.blink_detector.ear_closed_threshold and ear <= self.blink_detector.ear_closed_threshold:
            ear_color = (0, 0, 255)
        elif self.blink_detector.ear_closing_threshold and ear <= self.blink_detector.ear_closing_threshold:
            ear_color = (0, 165, 255)
        
        cv2.putText(frame, f"EAR: {ear:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
        y_offset += line_height
        
        # キャリブレーション状態
        if self.blink_detector.calibration_active:
            elapsed = time.time() - self.blink_detector.calibration_start_time
            remaining = self.blink_detector.calibration_duration - elapsed
            
            cv2.putText(frame, f"Calibrating: {remaining:.1f}s", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            calib_text = "Calibrated: YES" if stats['calibrated'] else "Calibrated: NO (Press C)"
            calib_color = (0, 255, 0) if stats['calibrated'] else (0, 0, 255)
            cv2.putText(frame, calib_text, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, calib_color, 2)
        
        # 操作方法（右下）
        instructions = [
            "[N] Normal session",
            "[D] Drowsy session",
            "[C] Calibrate",
            "[SPACE] Save",
            "[ESC] Quit"
        ]
        
        y_offset = h - 30 - (len(instructions) * 25)
        for instruction in instructions:
            cv2.putText(frame, instruction, 
                       (w - 250, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return frame
    
    def save_session(self):
        """現在のセッションを保存"""
        if not self.current_session or len(self.session_data) == 0:
            print("⚠️ 保存するデータがありません")
            return False
        
        session_id = self.current_session['session_id']
        label = self.current_session['label']
        label_name = self.current_session['label_name']
        
        # セッションファイルのパス
        session_file = os.path.join(
            self.data_dir, 
            'sessions', 
            f"{session_id}_{label_name}.json"
        )
        
        # セッションデータを準備
        session_export = {
            'session_id': session_id,
            'label': label,
            'label_name': label_name,
            'start_time': self.current_session['start_time'],
            'end_time': time.time(),
            'duration': time.time() - self.current_session['start_time'],
            'blink_count': len(self.session_data),
            'blinks': self.session_data
        }
        
        # JSON形式で保存
        with open(session_file, 'w') as f:
            json.dump(session_export, f, indent=2)
        
        print(f"\n✅ セッション保存成功: {session_file}")
        print(f"   瞬き数: {len(self.session_data)}")
        
        # シーケンスデータを生成・保存
        sequences_saved = self._save_sequences(session_id, label)
        
        # 統計を更新
        self.stats['total_sessions'] += 1
        self.stats['total_blinks'] += len(self.session_data)
        self.stats['total_sequences'] += sequences_saved
        
        if label == self.LABEL_NORMAL:
            self.stats['normal_sessions'] += 1
        else:
            self.stats['drowsy_sessions'] += 1
        
        # セッションをリセット
        self.current_session = None
        self.session_data = []
        self.collecting = False
        
        return True
    
    def _save_sequences(self, session_id, label):
        """
        シーケンスデータを生成・保存
        
        Args:
            session_id (str): セッションID
            label (int): ラベル
            
        Returns:
            int: 保存したシーケンス数
        """
        # 特徴量を抽出
        for i, blink_info in enumerate(self.session_data):
            # 既存のAPIに合わせてデータ形式を変換
            # t1 = 閉じ始め時刻、t2 = 完全閉眼時刻、t3 = 開き終わり時刻
            t3 = blink_info['timestamp']  # 開き終わり時刻
            t2 = t3 - blink_info['opening_time']  # 完全閉眼時刻
            t1 = t2 - blink_info['closing_time']  # 閉じ始め時刻
            
            blink_data = {
                't1': t1,
                't2': t2,
                't3': t3,
                'ear_min': blink_info['min_ear']
            }
            
            # 特徴量を抽出
            features = self.feature_extractor.extract_features(blink_data)
            
            if features is None:
                print(f"   ⚠️ 瞬き #{i+1} の特徴量抽出に失敗（異常値の可能性）")
        
        # バッチシーケンスを取得
        sequences_array, _ = self.feature_extractor.get_batch_sequences(normalize=False)
        
        if len(sequences_array) == 0:
            print(f"   ⚠️ シーケンスを生成できません（有効な瞬き数: {len(self.feature_extractor.blink_features)}）")
            print(f"   　　最低10回の有効な瞬きが必要です")
            return 0
        
        # JSON保存用にリストに変換
        sequences = []
        for seq in sequences_array:
            sequences.append({
                'features': seq.tolist(),
                'label': label
            })
        
        # シーケンスファイルのパス
        label_name = "normal" if label == self.LABEL_NORMAL else "drowsy"
        sequence_file = os.path.join(
            self.data_dir,
            'sequences',
            f"{session_id}_{label_name}_sequences.json"
        )
        
        # JSON形式で保存
        with open(sequence_file, 'w') as f:
            json.dump(sequences, f, indent=2)
        
        print(f"   ✅ シーケンス保存: {len(sequences)} 個")
        print(f"      ファイル: {sequence_file}")
        
        return len(sequences)
    
    def run(self):
        """メインループを実行"""
        if not self.initialize_camera():
            return
        
        print("\n" + "=" * 70)
        print("🚀 データ収集システム起動")
        print("=" * 70)
        print()
        print("操作方法:")
        print("  [N] - 正常状態のセッション開始")
        print("  [D] - 眠気状態のセッション開始")
        print("  [C] - キャリブレーション（最初に実行推奨）")
        print("  [SPACE] - 現在のセッションを保存")
        print("  [ESC] - 終了")
        print()
        print("=" * 70)
        print()
        
        # 最初にキャリブレーションを促す
        print("👉 まず[C]キーでキャリブレーションを実行してください")
        print()
        
        while True:
            ret, frame = self.camera.read()
            
            if not ret:
                print("⚠️ フレーム取得失敗")
                break
            
            # フレーム処理
            frame, blink_info = self.process_frame(frame)
            
            # UI描画
            frame = self.draw_ui(frame)
            
            # 表示
            cv2.imshow(self.window_name, frame)
            
            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                if self.current_session and self.collecting:
                    print("\n⚠️ セッション進行中です。保存しますか? (y/n)")
                    # ここでは自動で破棄
                    print("   セッションを破棄して終了します")
                break
            
            elif key == ord('c') or key == ord('C'):
                if not self.collecting:
                    self.start_calibration()
                else:
                    print("⚠️ セッション進行中はキャリブレーションできません")
            
            elif key == ord('n') or key == ord('N'):
                if not self.collecting:
                    self.start_session(self.LABEL_NORMAL)
                else:
                    print("⚠️ すでにセッション進行中です")
            
            elif key == ord('d') or key == ord('D'):
                if not self.collecting:
                    self.start_session(self.LABEL_DROWSY)
                else:
                    print("⚠️ すでにセッション進行中です")
            
            elif key == ord(' '):  # SPACE
                if self.collecting:
                    self.save_session()
                else:
                    print("⚠️ セッション進行中ではありません")
        
        # 終了処理
        self.cleanup()
    
    def cleanup(self):
        """リソースを解放"""
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        # 最終統計
        print("\n" + "=" * 70)
        print("📊 最終統計")
        print("=" * 70)
        print(f"総セッション数: {self.stats['total_sessions']}")
        print(f"  - 正常: {self.stats['normal_sessions']}")
        print(f"  - 眠気: {self.stats['drowsy_sessions']}")
        print(f"総瞬き数: {self.stats['total_blinks']}")
        print(f"総シーケンス数: {self.stats['total_sequences']}")
        print("=" * 70)
        print()
        print("データ収集システムを終了しました")


def main():
    """メイン関数"""
    collector = DrowsinessDataCollectorMediaPipe(
        data_dir="drowsiness_training_data",
        sequence_length=10
    )
    
    collector.run()


if __name__ == "__main__":
    main()