"""
眠気検出システム - データ収集プログラム（楕円フィッティング対応）

正常状態と眠気状態のデータを収集し、JSONファイルに保存します。
- MediaPipe Face Meshで顔・目を検出
- EAR（Eye Aspect Ratio）を計算
- 楕円フィッティングで目の形状を抽出
- 瞬き検出（4段階）
- 統計量 + 時系列データをJSONに保存

使い方:
    # 正常状態データ収集
    python drowsiness_data_collector_with_ellipse.py --label 0
    
    # 眠気状態データ収集（画面を1時間以上見た後）
    python drowsiness_data_collector_with_ellipse.py --label 1
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


class EllipseFitter:
    """楕円フィッティングクラス"""
    
    @staticmethod
    def fit(eye_landmarks):
        """
        目のランドマークから楕円パラメータを計算
        
        Args:
            eye_landmarks: 目のランドマーク [(x, y), ...]
            
        Returns:
            dict: 楕円パラメータ
        """
        try:
            if len(eye_landmarks) < 5:
                return None
            
            points = np.array(eye_landmarks, dtype=np.float32)
            
            # OpenCVの楕円フィッティング
            # 返り値: ((center_x, center_y), (axis1, axis2), angle)
            ellipse = cv2.fitEllipse(points)
            
            center = ellipse[0]
            axes = ellipse[1]
            angle = ellipse[2]
            
            # OpenCVのfitEllipseは、axis1とaxis2のどちらが大きいかで
            # 楕円の向きが決まる
            # 目は通常横長なので、大きい方が横幅（major_axis）になるはず
            
            # 長軸・短軸の決定
            if axes[0] >= axes[1]:
                major_axis = axes[0]
                minor_axis = axes[1]
                # angleはそのまま使用
                corrected_angle = angle
            else:
                # 軸が逆の場合、angleを90度回転
                major_axis = axes[1]
                minor_axis = axes[0]
                corrected_angle = angle + 90
            
            # 面積の計算
            area = np.pi * (major_axis / 2) * (minor_axis / 2)
            
            # 偏心率の計算
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0.0
            
            return {
                'center_x': float(center[0]),
                'center_y': float(center[1]),
                'major_axis': float(major_axis),
                'minor_axis': float(minor_axis),
                'area': float(area),
                'angle': float(corrected_angle),  # 補正された角度
                'eccentricity': float(eccentricity),
                # 元のfitEllipse結果も保持（描画用）
                'raw_axes': (float(axes[0]), float(axes[1])),
                'raw_angle': float(angle)
            }
            
        except Exception as e:
            return None


class BlinkDetectorWithEllipse:
    """
    瞬き検出器（楕円フィッティング + 時系列データ対応）
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
        self.current_blink_ellipse_timeseries = []
        
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
        
        # 両目の楕円パラメータを計算
        left_ellipse = EllipseFitter.fit(left_eye_landmarks)
        right_ellipse = EllipseFitter.fit(right_eye_landmarks)
        
        # 平均楕円パラメータ
        if left_ellipse and right_ellipse:
            avg_ellipse = {
                'major_axis': (left_ellipse['major_axis'] + right_ellipse['major_axis']) / 2,
                'minor_axis': (left_ellipse['minor_axis'] + right_ellipse['minor_axis']) / 2,
                'area': (left_ellipse['area'] + right_ellipse['area']) / 2,
                'angle': (left_ellipse['angle'] + right_ellipse['angle']) / 2,
                'eccentricity': (left_ellipse['eccentricity'] + right_ellipse['eccentricity']) / 2,
                'center_x': (left_ellipse['center_x'] + right_ellipse['center_x']) / 2,
                'center_y': (left_ellipse['center_y'] + right_ellipse['center_y']) / 2
            }
        elif left_ellipse:
            avg_ellipse = left_ellipse
        elif right_ellipse:
            avg_ellipse = right_ellipse
        else:
            avg_ellipse = None
        
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
            
            # 楕円時系列
            if avg_ellipse:
                ellipse_data = {
                    'timestamp': current_time,
                    'state': self._get_state_name(),
                    'center_x': avg_ellipse['center_x'],
                    'center_y': avg_ellipse['center_y'],
                    'major_axis': avg_ellipse['major_axis'],
                    'minor_axis': avg_ellipse['minor_axis'],
                    'area': avg_ellipse['area'],
                    'angle': avg_ellipse['angle'],
                    'eccentricity': avg_ellipse['eccentricity']
                }
                self.current_blink_ellipse_timeseries.append(ellipse_data)
        
        # 瞬き検出ロジック
        blink_info = self._detect_blink_state(ear, current_time)
        
        # 瞬き完了時の処理
        if blink_info is not None:
            # 統計量の計算
            if len(self.current_blink_ellipse_timeseries) > 0:
                ellipse_stats = self._calculate_ellipse_statistics()
                blink_info.update(ellipse_stats)
            else:
                blink_info.update({
                    'ellipse_major_axis_max': 0.0,
                    'ellipse_minor_axis_min': 0.0,
                    'ellipse_area_min': 0.0,
                    'ellipse_angle_change': 0.0,
                    'ellipse_eccentricity_max': 0.0
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
            blink_info['ellipse_timeseries'] = self.current_blink_ellipse_timeseries.copy()
            
            # クリア
            self.current_blink_ear_timeseries = []
            self.current_blink_ellipse_timeseries = []
            self.ear_history = []
        
        return blink_info
    
    def _calculate_ellipse_statistics(self):
        """楕円パラメータの統計量を計算"""
        if len(self.current_blink_ellipse_timeseries) == 0:
            return {
                'ellipse_major_axis_max': 0.0,
                'ellipse_minor_axis_min': 0.0,
                'ellipse_area_min': 0.0,
                'ellipse_angle_change': 0.0,
                'ellipse_eccentricity_max': 0.0
            }
        
        major_axes = [e['major_axis'] for e in self.current_blink_ellipse_timeseries]
        minor_axes = [e['minor_axis'] for e in self.current_blink_ellipse_timeseries]
        areas = [e['area'] for e in self.current_blink_ellipse_timeseries]
        angles = [e['angle'] for e in self.current_blink_ellipse_timeseries]
        eccentricities = [e['eccentricity'] for e in self.current_blink_ellipse_timeseries]
        
        return {
            'ellipse_major_axis_max': float(max(major_axes)),
            'ellipse_minor_axis_min': float(min(minor_axes)),
            'ellipse_area_min': float(min(areas)),
            'ellipse_angle_change': float(max(angles) - min(angles)),
            'ellipse_eccentricity_max': float(max(eccentricities))
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


class DataCollectorWithEllipse:
    """データ収集クラス（楕円フィッティング対応）"""
    
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
        self.blink_detector = BlinkDetectorWithEllipse()
        
        # セッション情報
        self.session_data = {
            'session_id': None,
            'label': None,
            'start_time': None,
            'end_time': None,
            'duration': 0.0,
            'total_blinks': 0,
            'valid_blinks': 0,
            'blinks': []
        }
        
        self.blink_counter = 0
        self.session_start_time = None
    
    def start_session(self, label):
        """セッション開始"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_str = "normal" if label == 0 else "drowsy"
        
        self.session_data['session_id'] = f"{timestamp}_{label_str}"
        self.session_data['label'] = label
        self.session_data['start_time'] = datetime.now().isoformat()
        self.session_data['blinks'] = []
        self.blink_counter = 0
        self.session_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"セッション開始: {self.session_data['session_id']}")
        print(f"ラベル: {label_str}")
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
                'ellipse_major_axis_max': blink_info['ellipse_major_axis_max'],
                'ellipse_minor_axis_min': blink_info['ellipse_minor_axis_min'],
                'ellipse_area_min': blink_info['ellipse_area_min'],
                'ellipse_angle_change': blink_info['ellipse_angle_change'],
                'ellipse_eccentricity_max': blink_info['ellipse_eccentricity_max']
            },
            'ear_timeseries': blink_info['ear_timeseries'],
            'ellipse_timeseries': blink_info['ellipse_timeseries']
        }
        
        self.session_data['blinks'].append(blink_data)
        self.session_data['total_blinks'] += 1
        
        # 簡易的な有効性チェック
        if self._is_valid_blink(blink_data['statistics']):
            self.session_data['valid_blinks'] += 1
        
        print(f"瞬き検出 #{self.blink_counter}: "
              f"係数={blink_info['blink_coefficient']:.2f}, "
              f"Tc={blink_info['closing_time']*1000:.0f}ms, "
              f"To={blink_info['opening_time']*1000:.0f}ms")
    
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
        
        # 楕円フィッティング
        left_ellipse = EllipseFitter.fit(left_eye)
        right_ellipse = EllipseFitter.fit(right_eye)
        
        # 平均楕円パラメータ
        if left_ellipse and right_ellipse:
            avg_ellipse = {
                'major_axis': (left_ellipse['major_axis'] + right_ellipse['major_axis']) / 2,
                'minor_axis': (left_ellipse['minor_axis'] + right_ellipse['minor_axis']) / 2,
                'area': (left_ellipse['area'] + right_ellipse['area']) / 2
            }
        elif left_ellipse:
            avg_ellipse = {
                'major_axis': left_ellipse['major_axis'],
                'minor_axis': left_ellipse['minor_axis'],
                'area': left_ellipse['area']
            }
        elif right_ellipse:
            avg_ellipse = {
                'major_axis': right_ellipse['major_axis'],
                'minor_axis': right_ellipse['minor_axis'],
                'area': right_ellipse['area']
            }
        else:
            avg_ellipse = None
        
        # 瞬き検出
        blink_info = self.blink_detector.detect(avg_ear, left_eye, right_eye)
        
        # 可視化
        frame = self._draw_visualization(frame, left_eye, right_eye, 
                                        left_ellipse, right_ellipse,
                                        avg_ear, avg_ellipse)
        
        return frame, blink_info, avg_ear, avg_ellipse
    
    def _draw_visualization(self, frame, left_eye, right_eye, 
                           left_ellipse, right_ellipse, avg_ear, avg_ellipse):
        """可視化"""
        # 目のランドマークを描画
        for point in left_eye + right_eye:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        
        # 左目の楕円を描画（元のfitEllipse結果を使用）
        if left_ellipse:
            try:
                center = (int(left_ellipse['center_x']), int(left_ellipse['center_y']))
                
                # 元のfitEllipse結果を使用（こちらが正しい）
                axes = (int(left_ellipse['raw_axes'][0]/2), 
                       int(left_ellipse['raw_axes'][1]/2))
                angle = int(left_ellipse['raw_angle'])
                
                cv2.ellipse(frame, center, axes, angle, 0, 360, (255, 0, 0), 2)
            except Exception as e:
                pass
        
        # 右目の楕円を描画
        if right_ellipse:
            try:
                center = (int(right_ellipse['center_x']), int(right_ellipse['center_y']))
                axes = (int(right_ellipse['raw_axes'][0]/2), 
                       int(right_ellipse['raw_axes'][1]/2))
                angle = int(right_ellipse['raw_angle'])
                
                cv2.ellipse(frame, center, axes, angle, 0, 360, (255, 0, 0), 2)
            except Exception as e:
                pass
        
        # 情報表示
        y_offset = 30
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if avg_ellipse:
            y_offset += 30
            cv2.putText(frame, f"Major Axis: {avg_ellipse['major_axis']:.1f}px",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            y_offset += 25
            cv2.putText(frame, f"Minor Axis: {avg_ellipse['minor_axis']:.1f}px",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            y_offset += 25
            cv2.putText(frame, f"Area: {avg_ellipse['area']:.1f}px2",
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


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='眠気検出データ収集プログラム')
    parser.add_argument('--label', type=int, required=True, choices=[0, 1],
                       help='ラベル: 0=正常状態, 1=眠気状態')
    parser.add_argument('--output-dir', type=str, default='data/sessions',
                       help='出力ディレクトリ（デフォルト: data/sessions）')
    args = parser.parse_args()
    
    # データ収集器の初期化
    collector = DataCollectorWithEllipse(output_dir=args.output_dir)
    
    # セッション開始
    collector.start_session(label=args.label)
    
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
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ フレームの取得に失敗しました")
                break
            
            # フレーム処理
            processed_frame, blink_info, avg_ear, avg_ellipse = collector.process_frame(frame)
            
            # 瞬き検出時
            if blink_info is not None:
                collector.add_blink(blink_info)
            
            # 画面表示
            cv2.imshow('Data Collection', processed_frame)
            
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