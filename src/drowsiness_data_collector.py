"""
眠気推定用データ収集システム
Drowsiness Data Collection System

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

# 既存モジュールのインポートを試みる
try:
    from src.blink_detector import BlinkDetector
    from src.blink_feature_extractor import BlinkFeatureExtractor
except ImportError as e:
    print(f"⚠️ モジュールのインポートエラー: {e}")
    print("   src/blink_detector.py と src/blink_feature_extractor.py が必要です")
    print("   プロジェクトルートから実行してください: python -m src.drowsiness_data_collector")


class DrowsinessDataCollector:
    """
    眠気推定用データ収集システム
    
    機能:
    - リアルタイム瞬き検出と特徴量抽出
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
        
        # 瞬き検出器
        self.blink_detector = BlinkDetector()
        
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
        self.window_name = "眠気データ収集システム"
        
        print("=" * 70)
        print("🎯 眠気推定用データ収集システム初期化完了")
        print("=" * 70)
        print(f"📁 データ保存先: {self.data_dir}")
        print(f"📊 シーケンス長: {self.sequence_length}")
    
    def initialize_camera(self, camera_id=0):
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
            
            # キャリブレーション
            print("\n🎯 個人キャリブレーション開始（5秒間）")
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
                            cv2.imshow(self.window_name, display_frame)
                            cv2.waitKey(1)
            
            print("✅ キャリブレーション完了")
            print(f"   カメラ解像度: {self.camera_width}x{self.camera_height}")
            print(f"   FPS: {self.fps}")
            
            return True
            
        except Exception as e:
            print(f"❌ カメラ初期化エラー: {e}")
            return False
    
    def start_session(self, label: int, session_name: Optional[str] = None):
        """
        データ収集セッションを開始
        
        Args:
            label (int): セッションのラベル (0: 正常, 1: 眠気)
            session_name (str): セッション名（省略時は自動生成）
            
        Returns:
            bool: 成功したかどうか
        """
        if self.collecting:
            print("⚠️ 既にデータ収集中です")
            return False
        
        if self.camera is None or not self.camera.isOpened():
            print("❌ カメラが初期化されていません")
            return False
        
        # セッション情報
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            label_str = "normal" if label == self.LABEL_NORMAL else "drowsy"
            session_name = f"{label_str}_{timestamp}"
        
        self.current_session = {
            'session_name': session_name,
            'label': label,
            'start_time': time.time(),
            'end_time': None,
            'blink_count': 0,
            'sequence_count': 0
        }
        
        self.session_data = []
        self.session_label = label
        self.collecting = True
        
        label_text = "正常状態" if label == self.LABEL_NORMAL else "眠気状態"
        print("\n" + "=" * 70)
        print(f"🚀 データ収集開始")
        print("=" * 70)
        print(f"📝 セッション名: {session_name}")
        print(f"🏷️  ラベル: {label_text} ({label})")
        print(f"⏱️  開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n操作方法:")
        print("  [SPACE] - セッション終了・保存")
        print("  [ESC]   - セッション破棄・終了")
        print("  [N]     - 正常状態に切替")
        print("  [D]     - 眠気状態に切替")
        print("=" * 70)
        
        return True
    
    def stop_session(self, save=True):
        """
        データ収集セッションを終了
        
        Args:
            save (bool): データを保存するか
            
        Returns:
            bool: 成功したかどうか
        """
        if not self.collecting:
            print("⚠️ データ収集中ではありません")
            return False
        
        self.collecting = False
        self.current_session['end_time'] = time.time()
        
        duration = self.current_session['end_time'] - self.current_session['start_time']
        
        print("\n" + "=" * 70)
        print("📊 セッション終了")
        print("=" * 70)
        print(f"⏱️  収集時間: {duration:.1f}秒")
        print(f"👁️  瞬き回数: {self.current_session['blink_count']}")
        print(f"📦 シーケンス数: {self.current_session['sequence_count']}")
        
        if save and len(self.session_data) > 0:
            success = self._save_session()
            if success:
                print("✅ データ保存成功")
                return True
            else:
                print("❌ データ保存失敗")
                return False
        elif not save:
            print("⚠️ データを破棄しました")
            return False
        else:
            print("⚠️ 保存するデータがありません")
            return False
    
    def collect_frame(self, frame):
        """
        1フレームのデータを収集
        
        Args:
            frame: カメラフレーム
            
        Returns:
            tuple: (処理済みフレーム, 瞬き検出結果)
        """
        if not self.collecting:
            return frame, None
        
        # 顔検出（OpenCV Haar Cascade使用）
        face_rect = self.blink_detector.detect_face(frame)
        
        ear = None
        blink_detected = False
        blink_state = None
        
        if face_rect is not None:
            # 瞬き検出
            blink_detected, ear, blink_state = self.blink_detector.detect_blink(frame, face_rect)
        
        # 瞬きが検出された場合
        if blink_detected:
            blink_data = self._get_blink_data()
            
            if blink_data is not None:
                # 特徴量抽出
                features = self.feature_extractor.extract_features(blink_data)
                
                if features is not None:
                    # データ記録
                    data_point = {
                        'timestamp': time.time(),
                        'features': features.tolist(),
                        'blink_data': blink_data,
                        'label': self.session_label
                    }
                    self.session_data.append(data_point)
                    self.current_session['blink_count'] += 1
                    
                    # シーケンスデータの取得を試みる
                    sequence = self.feature_extractor.get_sequence(normalize=False)
                    if sequence is not None:
                        self.current_session['sequence_count'] += 1
        
        # 可視化
        if self.show_visualization:
            # BlinkDetectorの詳細情報を描画
            self.blink_detector.draw_debug_info(frame, face_rect)
            
            # 追加の統計情報を描画
            frame = self._draw_additional_info(frame, ear, blink_detected, face_rect)
        
        return frame, blink_detected
    
    def _get_blink_data(self) -> Optional[Dict]:
        """
        最新の瞬きデータを取得
        
        Returns:
            Dict: 瞬きデータ（t1, t2, t3, ear_min）
        """
        if len(self.blink_detector.blink_details) == 0:
            return None
        
        latest_blink = self.blink_detector.blink_details[-1]
        
        # 必要なキーが存在するか確認
        required_keys = ['t1', 't2', 't3', 'ear_min']
        if not all(key in latest_blink for key in required_keys):
            return None
        
        return {
            't1': latest_blink['t1'],
            't2': latest_blink['t2'],
            't3': latest_blink['t3'],
            'ear_min': latest_blink['ear_min']
        }
    
    def _draw_additional_info(self, frame, ear, blink_detected, face_rect):
        """
        セッション情報とEARグラフを描画
        
        Args:
            frame: 元フレーム
            ear: EAR値
            blink_detected: 瞬き検出フラグ
            face_rect: 顔の矩形
            
        Returns:
            処理済みフレーム
        """
        h, w = frame.shape[:2]
        
        # 右上にセッション情報パネルを描画
        panel_x = w - 280
        panel_y = 10
        panel_w = 270
        panel_h = 200
        
        # 半透明の背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # セッション情報
        y_offset = panel_y + 30
        line_height = 28
        
        # ラベル表示（大きく目立つように）
        label_text = "NORMAL" if self.session_label == self.LABEL_NORMAL else "DROWSY"
        label_color = (0, 255, 0) if self.session_label == self.LABEL_NORMAL else (0, 140, 255)
        
        cv2.putText(frame, f"[{label_text}]", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)
        y_offset += line_height + 5
        
        # 収集統計
        cv2.putText(frame, f"Blinks: {self.current_session['blink_count']}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        cv2.putText(frame, f"Sequences: {self.current_session['sequence_count']}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        # 瞬き検出状態
        if blink_detected:
            cv2.putText(frame, "BLINK!", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += line_height
        
        # 顔検出状態
        if face_rect is not None:
            cv2.putText(frame, "Face: OK", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Face: LOST", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 下部にEARグラフを描画
        self._draw_ear_graph(frame, ear)
        
        # 下部中央に操作ガイドを表示
        self._draw_controls_guide(frame)
        
        return frame
    
    def _draw_ear_graph(self, frame, current_ear):
        """
        EARグラフを描画
        
        Args:
            frame: 元フレーム
            current_ear: 現在のEAR値
        """
        h, w = frame.shape[:2]
        
        # グラフの設定
        graph_x = 10
        graph_y = h - 150
        graph_w = 300
        graph_h = 100
        
        # 背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # 枠線
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (100, 100, 100), 2)
        
        # タイトル
        cv2.putText(frame, "EAR History", (graph_x + 5, graph_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # EAR履歴を取得
        ear_values = list(self.blink_detector.ear_values)
        
        if len(ear_values) > 1:
            # 最大100ポイントを表示
            max_points = min(100, len(ear_values))
            ear_subset = ear_values[-max_points:]
            
            # 正規化（0.1-0.5の範囲をグラフの高さにマッピング）
            min_ear = 0.1
            max_ear = 0.5
            
            points = []
            for i, ear in enumerate(ear_subset):
                x = graph_x + int((i / max_points) * graph_w)
                # EAR値をグラフの高さにマッピング（上下反転）
                normalized = (ear - min_ear) / (max_ear - min_ear)
                normalized = max(0, min(1, normalized))  # 0-1の範囲にクリップ
                y = graph_y + graph_h - int(normalized * graph_h)
                points.append((x, y))
            
            # 線を描画
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)
            
            # 閾値ラインを描画
            if self.blink_detector.ear_closed_threshold is not None:
                threshold = self.blink_detector.ear_closed_threshold
                normalized_threshold = (threshold - min_ear) / (max_ear - min_ear)
                threshold_y = graph_y + graph_h - int(normalized_threshold * graph_h)
                cv2.line(frame, (graph_x, threshold_y), (graph_x + graph_w, threshold_y), 
                        (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Threshold: {threshold:.2f}", 
                           (graph_x + 5, threshold_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 現在のEAR値を表示
        if current_ear is not None:
            cv2.putText(frame, f"Current: {current_ear:.3f}", 
                       (graph_x + 5, graph_y + graph_h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _draw_controls_guide(self, frame):
        """
        操作ガイドを描画
        
        Args:
            frame: 元フレーム
        """
        h, w = frame.shape[:2]
        
        # 中央下部に表示
        guide_x = w // 2 - 200
        guide_y = h - 35
        
        # 背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (guide_x - 5, guide_y - 25), (guide_x + 400, guide_y + 5), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # テキスト
        controls_text = "[N]Normal  [D]Drowsy  [SPACE]Save  [ESC]Exit"
        cv2.putText(frame, controls_text, (guide_x, guide_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _save_session(self) -> bool:
        """
        セッションデータを保存
        
        Returns:
            bool: 成功したかどうか
        """
        try:
            session_name = self.current_session['session_name']
            
            # セッション情報を保存（JSON）
            session_info_path = os.path.join(
                self.data_dir, 'sessions', f"{session_name}_info.json"
            )
            
            session_info = {
                'session_name': session_name,
                'label': self.session_label,
                'label_name': 'normal' if self.session_label == self.LABEL_NORMAL else 'drowsy',
                'start_time': self.current_session['start_time'],
                'end_time': self.current_session['end_time'],
                'duration': self.current_session['end_time'] - self.current_session['start_time'],
                'blink_count': self.current_session['blink_count'],
                'sequence_count': self.current_session['sequence_count'],
                'data_points': len(self.session_data)
            }
            
            with open(session_info_path, 'w') as f:
                json.dump(session_info, f, indent=2)
            
            # 瞬きデータを保存（CSV）
            csv_path = os.path.join(
                self.data_dir, 'sessions', f"{session_name}_blinks.csv"
            )
            
            with open(csv_path, 'w', newline='') as f:
                fieldnames = [
                    'timestamp', 'label',
                    'blink_coefficient', 'tc', 'to', 'interval', 'ear_min', 'total_duration',
                    't1', 't2', 't3'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for data_point in self.session_data:
                    features = data_point['features']
                    blink_data = data_point['blink_data']
                    
                    row = {
                        'timestamp': data_point['timestamp'],
                        'label': data_point['label'],
                        'blink_coefficient': features[0],
                        'tc': features[1],
                        'to': features[2],
                        'interval': features[3],
                        'ear_min': features[4],
                        'total_duration': features[5],
                        't1': blink_data['t1'],
                        't2': blink_data['t2'],
                        't3': blink_data['t3']
                    }
                    writer.writerow(row)
            
            # シーケンスデータを保存（NumPy）
            sequences, raw_data = self.feature_extractor.get_batch_sequences(normalize=False)
            
            if len(sequences) > 0:
                seq_path = os.path.join(
                    self.data_dir, 'sequences', f"{session_name}_sequences.npz"
                )
                
                labels = np.full(len(sequences), self.session_label, dtype=np.int64)
                
                np.savez(seq_path,
                        sequences=sequences,
                        labels=labels,
                        session_name=session_name)
            
            # 統計更新
            self.stats['total_sessions'] += 1
            if self.session_label == self.LABEL_NORMAL:
                self.stats['normal_sessions'] += 1
            else:
                self.stats['drowsy_sessions'] += 1
            self.stats['total_blinks'] += self.current_session['blink_count']
            self.stats['total_sequences'] += self.current_session['sequence_count']
            
            # 統計を保存
            self._save_statistics()
            
            print(f"💾 保存完了:")
            print(f"   セッション情報: {session_info_path}")
            print(f"   瞬きデータ: {csv_path}")
            if len(sequences) > 0:
                print(f"   シーケンスデータ: {seq_path} ({len(sequences)} sequences)")
            
            return True
            
        except Exception as e:
            print(f"❌ 保存エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_statistics(self):
        """統計情報を保存"""
        stats_path = os.path.join(self.data_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def load_statistics(self):
        """統計情報を読み込み"""
        stats_path = os.path.join(self.data_dir, 'statistics.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
    
    def print_statistics(self):
        """統計情報を表示"""
        print("\n" + "=" * 70)
        print("📊 データ収集統計")
        print("=" * 70)
        print(f"総セッション数: {self.stats['total_sessions']}")
        print(f"  正常状態: {self.stats['normal_sessions']}")
        print(f"  眠気状態: {self.stats['drowsy_sessions']}")
        print(f"総瞬き数: {self.stats['total_blinks']}")
        print(f"総シーケンス数: {self.stats['total_sequences']}")
        print("=" * 70)
    
    def run_interactive(self):
        """
        インタラクティブなデータ収集を実行
        """
        if not self.initialize_camera():
            return
        
        print("\n" + "=" * 70)
        print("🎮 インタラクティブモード")
        print("=" * 70)
        print("操作方法:")
        print("  [N] - 正常状態のデータ収集開始")
        print("  [D] - 眠気状態のデータ収集開始")
        print("  [SPACE] - 現在のセッション終了・保存")
        print("  [ESC] - セッション破棄またはプログラム終了")
        print("  [S] - 統計情報表示")
        print("=" * 70)
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ フレーム取得失敗")
                    break
                
                # フレーム処理
                frame, _ = self.collect_frame(frame)
                
                # 表示
                cv2.imshow(self.window_name, frame)
                
                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('n') or key == ord('N'):
                    if not self.collecting:
                        self.start_session(self.LABEL_NORMAL)
                    else:
                        print("⚠️ 既にセッション実行中です")
                
                elif key == ord('d') or key == ord('D'):
                    if not self.collecting:
                        self.start_session(self.LABEL_DROWSY)
                    else:
                        print("⚠️ 既にセッション実行中です")
                
                elif key == ord(' '):  # SPACE
                    if self.collecting:
                        self.stop_session(save=True)
                
                elif key == 27:  # ESC
                    if self.collecting:
                        print("\n⚠️ セッションを破棄しますか? (y/n)")
                        # 一時的に待機（簡易的な確認）
                        self.stop_session(save=False)
                    else:
                        print("\n👋 終了します")
                        break
                
                elif key == ord('s') or key == ord('S'):
                    self.print_statistics()
        
        finally:
            if self.collecting:
                self.stop_session(save=True)
            
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 70)
            print("最終統計:")
            self.print_statistics()
            print("=" * 70)


# メイン実行
if __name__ == "__main__":
    print("=" * 70)
    print("眠気推定用データ収集システム")
    print("=" * 70)
    
    collector = DrowsinessDataCollector()
    collector.load_statistics()
    collector.print_statistics()
    
    print("\nインタラクティブモードを開始します...")
    print("正常状態と眠気状態の両方のデータを収集してください。")
    
    collector.run_interactive()