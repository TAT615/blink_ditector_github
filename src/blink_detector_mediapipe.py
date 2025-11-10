"""
瞬き検出モジュール（MediaPipe版）
MediaPipe Face Meshを使用して高精度な顔・目のランドマーク検出と瞬き検出を行います。
"""

import numpy as np
import cv2
import time
import mediapipe as mp
from collections import deque


class BlinkDetectorMediaPipe:
    """
    MediaPipe Face Meshを使用した瞬き検出器
    
    特徴:
    - 顔検出: MediaPipe Face Mesh
    - 目検出: 478個の高精度ランドマーク
    - EAR (Eye Aspect Ratio) による瞬き検出
    - 個人キャリブレーション機能
    """
    
    # 瞬き状態の定義（4段階）
    BLINK_STATE_OPEN = 0          # 完全開眼
    BLINK_STATE_CLOSING = 1       # 閉眼途中
    BLINK_STATE_CLOSED = 2        # 完全閉眼
    BLINK_STATE_OPENING = 3       # 開眼途中
    
    # MediaPipe Face Meshの目のランドマークインデックス
    # 左目のランドマーク（6点）
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    # 右目のランドマーク（6点）
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    def __init__(self, buffer_size=300, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初期化
        
        Args:
            buffer_size (int): データバッファサイズ
            min_detection_confidence (float): 顔検出の最小信頼度
            min_tracking_confidence (float): トラッキングの最小信頼度
        """
        self.buffer_size = buffer_size
        self.blink_count = 0
        self.blink_times = deque(maxlen=buffer_size)
        self.blink_rates = deque(maxlen=buffer_size)
        self.ear_values = deque(maxlen=buffer_size)
        self.last_blink_time = time.time()
        
        # 瞬き時間の記録用
        self.blink_durations = deque(maxlen=buffer_size)
        self.blink_details = deque(maxlen=buffer_size)
        
        # 瞬き状態の管理
        self.blink_state = self.BLINK_STATE_OPEN
        self.current_blink_start = None
        self.current_blink_closed = None
        self.current_blink_min_ear = 1.0
        self.previous_ear = None
        
        # MediaPipe Face Meshの初期化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # キャリブレーション用パラメータ
        self.calibration_active = False
        self.calibration_start_time = None
        self.calibration_duration = 5.0
        
        # ベースラインEAR値の収集
        self.baseline_ear_values = deque(maxlen=150)
        self.min_ear_values = deque(maxlen=50)
        
        # 個人のダイナミックレンジ
        self.ear_open_max = None
        self.ear_closed_min = None
        self.ear_dynamic_range = None
        
        # 4段階の個人閾値
        self.ear_open_threshold = None
        self.ear_opening_threshold = None
        self.ear_closing_threshold = None
        self.ear_closed_threshold = None
        
        # デフォルト閾値
        self.default_open_threshold = 0.30
        self.default_closed_threshold = 0.20
        
        # 瞬き検出のための状態管理
        self.face_lost_frames = 0
        self.MAX_FACE_LOST_FRAMES = 2
        self.last_valid_ear = None
        
        # 瞬き率の計算用
        self.blink_window = 60
        self.recent_blinks = deque(maxlen=buffer_size)
        
        # 目の表面積記録用
        self.left_eye_areas = deque(maxlen=buffer_size)
        self.right_eye_areas = deque(maxlen=buffer_size)
        self.total_eye_areas = deque(maxlen=buffer_size)
        
        # キャリブレーション状態の記録
        self.calibration_log = []

    def start_calibration(self):
        """キャリブレーション開始"""
        self.calibration_active = True
        self.calibration_start_time = time.time()
        self.baseline_ear_values.clear()
        self.min_ear_values.clear()
        print("🎯 個人キャリブレーション開始: 5秒間リラックスして自然に瞬きしてください")

    def update_calibration(self, ear):
        """
        キャリブレーション中のEAR値更新
        
        Args:
            ear (float): 現在のEAR値
            
        Returns:
            bool: キャリブレーション完了したかどうか
        """
        if not self.calibration_active or self.calibration_start_time is None:
            return False
            
        elapsed = time.time() - self.calibration_start_time
        
        if elapsed < self.calibration_duration:
            # EAR値を収集
            self.baseline_ear_values.append(ear)
            return False
        else:
            # キャリブレーション完了
            self._finalize_calibration()
            return True

    def _finalize_calibration(self):
        """キャリブレーションの完了処理"""
        if len(self.baseline_ear_values) < 30:
            print("⚠️ キャリブレーションデータ不足。デフォルト値を使用します。")
            self._use_default_thresholds()
            self.calibration_active = False
            return
            
        # 開眼時のEAR値（75パーセンタイル）
        self.ear_open_max = np.percentile(list(self.baseline_ear_values), 75)
        
        # 閉眼時のEAR値（25パーセンタイル）
        self.ear_closed_min = np.percentile(list(self.baseline_ear_values), 25)
        
        # ダイナミックレンジ
        self.ear_dynamic_range = self.ear_open_max - self.ear_closed_min
        
        if self.ear_dynamic_range < 0.05:
            print("⚠️ ダイナミックレンジが小さすぎます。デフォルト値を使用します。")
            self._use_default_thresholds()
        else:
            # 個人化された閾値を設定
            self.ear_open_threshold = self.ear_closed_min + 0.75 * self.ear_dynamic_range
            self.ear_opening_threshold = self.ear_closed_min + 0.50 * self.ear_dynamic_range
            self.ear_closing_threshold = self.ear_closed_min + 0.35 * self.ear_dynamic_range
            self.ear_closed_threshold = self.ear_closed_min + 0.15 * self.ear_dynamic_range
            
            print(f"✅ キャリブレーション完了")
            print(f"   開眼EAR: {self.ear_open_max:.3f}")
            print(f"   閉眼EAR: {self.ear_closed_min:.3f}")
            print(f"   ダイナミックレンジ: {self.ear_dynamic_range:.3f}")
        
        self.calibration_active = False

    def _use_default_thresholds(self):
        """デフォルトの閾値を使用"""
        self.ear_open_threshold = self.default_open_threshold
        self.ear_closed_threshold = self.default_closed_threshold
        self.ear_opening_threshold = 0.25
        self.ear_closing_threshold = 0.23

    def detect_face_and_landmarks(self, frame):
        """
        MediaPipe Face Meshで顔とランドマークを検出
        
        Args:
            frame: BGR画像フレーム
            
        Returns:
            landmarks: 顔のランドマーク（正規化座標）、検出失敗時はNone
        """
        # BGRからRGBに変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipeで処理
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # 最初の顔のランドマークを返す
            self.face_lost_frames = 0
            return results.multi_face_landmarks[0]
        else:
            self.face_lost_frames += 1
            if self.face_lost_frames > self.MAX_FACE_LOST_FRAMES * 2:
                self._reset_detection_state()
            return None
    
    def _reset_detection_state(self):
        """検出状態をリセット"""
        self.face_lost_frames = 0
        self.last_valid_ear = None

    def calculate_ear_from_landmarks(self, landmarks, frame_shape):
        """
        ランドマークからEARを計算
        
        Args:
            landmarks: MediaPipeの顔ランドマーク
            frame_shape: フレームの形状 (height, width, channels)
            
        Returns:
            float: 両目の平均EAR値
        """
        h, w = frame_shape[:2]
        
        # ランドマークをピクセル座標に変換
        def get_landmarks_points(indices):
            points = []
            for idx in indices:
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append((x, y))
            return points
        
        # 左目と右目のランドマーク取得
        left_eye_points = get_landmarks_points(self.LEFT_EYE_INDICES)
        right_eye_points = get_landmarks_points(self.RIGHT_EYE_INDICES)
        
        # 各目のEAR計算
        left_ear = self._calculate_single_eye_ear(left_eye_points)
        right_ear = self._calculate_single_eye_ear(right_eye_points)
        
        # 平均EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        return avg_ear
    
    def _calculate_single_eye_ear(self, eye_points):
        """
        単一の目のEARを計算
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_points: 目の6点のランドマーク座標 [(x1,y1), (x2,y2), ...]
            
        Returns:
            float: EAR値
        """
        # 距離計算のヘルパー関数
        def euclidean_distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # MediaPipeの目のランドマーク配置:
        # [0]: 左端
        # [1]: 上部中央の左
        # [2]: 上部中央の右
        # [3]: 右端
        # [4]: 下部中央の右
        # [5]: 下部中央の左
        
        # 垂直距離
        vertical_1 = euclidean_distance(eye_points[1], eye_points[5])  # 上左 - 下左
        vertical_2 = euclidean_distance(eye_points[2], eye_points[4])  # 上右 - 下右
        
        # 水平距離
        horizontal = euclidean_distance(eye_points[0], eye_points[3])  # 左端 - 右端
        
        # EAR計算
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        
        return ear

    def detect_blink(self, frame, face_rect=None):
        """
        瞬きを検出（MediaPipe版）
        
        Args:
            frame: BGR画像フレーム
            face_rect: 互換性のため残しているが使用しない
            
        Returns:
            dict or None: 瞬きが完了した場合、瞬き情報を含む辞書を返す
        """
        # 顔とランドマークを検出
        landmarks = self.detect_face_and_landmarks(frame)
        
        if landmarks is None:
            return None
        
        # EAR計算
        current_ear = self.calculate_ear_from_landmarks(landmarks, frame.shape)
        
        # キャリブレーション中の場合
        if self.calibration_active:
            self.update_calibration(current_ear)
            return None
        
        # 閾値が設定されていない場合はデフォルト使用
        if self.ear_open_threshold is None:
            self._use_default_thresholds()
        
        # EAR値を記録
        self.ear_values.append(current_ear)
        
        # 4段階状態遷移検出
        blink_info = self._detect_4_stage_blink(current_ear)
        
        # 前回のEAR値を更新
        self.previous_ear = current_ear
        
        return blink_info

    def _detect_4_stage_blink(self, current_ear):
        """
        4段階の瞬き状態遷移を検出
        
        Args:
            current_ear (float): 現在のEAR値
            
        Returns:
            dict or None: 完全な瞬きが検出された場合、詳細情報を返す
        """
        current_time = time.time()
        state_changed = False
        
        # 状態遷移の判定
        if self.blink_state == self.BLINK_STATE_OPEN:
            # 開眼 → 閉眼途中
            if current_ear < self.ear_open_threshold:
                self.blink_state = self.BLINK_STATE_CLOSING
                self.current_blink_start = current_time
                self.current_blink_min_ear = current_ear
                state_changed = True
                
        elif self.blink_state == self.BLINK_STATE_CLOSING:
            # 最小EAR更新
            if current_ear < self.current_blink_min_ear:
                self.current_blink_min_ear = current_ear
            
            # 閉眼途中 → 完全閉眼
            if current_ear <= self.ear_closed_threshold:
                self.blink_state = self.BLINK_STATE_CLOSED
                self.current_blink_closed = current_time
                state_changed = True
            # 閉眼途中 → 開眼（不完全な瞬き）
            elif current_ear > self.ear_opening_threshold:
                self.blink_state = self.BLINK_STATE_OPEN
                self.current_blink_start = None
                self.current_blink_closed = None
                state_changed = True
                
        elif self.blink_state == self.BLINK_STATE_CLOSED:
            # 最小EAR更新
            if current_ear < self.current_blink_min_ear:
                self.current_blink_min_ear = current_ear
            
            # 完全閉眼 → 開眼途中
            if current_ear > self.ear_closed_threshold:
                self.blink_state = self.BLINK_STATE_OPENING
                state_changed = True
                
        elif self.blink_state == self.BLINK_STATE_OPENING:
            # 開眼途中 → 完全開眼（瞬き完了）
            # ear_open_thresholdは厳しすぎるので、ear_opening_thresholdを使用
            if current_ear >= self.ear_opening_threshold:
                blink_end = current_time
                
                # 瞬き時間パラメータの計算
                if (self.current_blink_start is not None and 
                    self.current_blink_closed is not None):
                    
                    closing_time = self.current_blink_closed - self.current_blink_start
                    opening_time = blink_end - self.current_blink_closed
                    total_duration = blink_end - self.current_blink_start
                    
                    # 瞬き係数の計算
                    blink_coefficient = opening_time / closing_time if closing_time > 0 else 0
                    
                    # 瞬き情報を作成
                    blink_info = {
                        'timestamp': blink_end,
                        'closing_time': closing_time,
                        'opening_time': opening_time,
                        'total_duration': total_duration,
                        'blink_coefficient': blink_coefficient,
                        'min_ear': self.current_blink_min_ear,
                        'interval': blink_end - self.last_blink_time
                    }
                    
                    # 記録を更新
                    self.blink_count += 1
                    self.blink_times.append(blink_end)
                    self.blink_durations.append(total_duration)
                    self.blink_details.append(blink_info)
                    self.last_blink_time = blink_end
                    
                    # 状態をリセット
                    self.blink_state = self.BLINK_STATE_OPEN
                    self.current_blink_start = None
                    self.current_blink_closed = None
                    self.current_blink_min_ear = 1.0
                    
                    return blink_info
                
                # エラー回復
                self.blink_state = self.BLINK_STATE_OPEN
                self.current_blink_start = None
                self.current_blink_closed = None
                state_changed = True
        
        return None

    def get_blink_rate(self, window_seconds=60):
        """
        指定期間内の瞬き率を計算
        
        Args:
            window_seconds (int): 計算期間（秒）
            
        Returns:
            float: 1分あたりの瞬き回数
        """
        current_time = time.time()
        recent_count = sum(1 for t in self.blink_times 
                          if current_time - t <= window_seconds)
        
        # 1分あたりに正規化
        rate = (recent_count / window_seconds) * 60
        return rate

    def get_statistics(self):
        """
        瞬き統計情報を取得
        
        Returns:
            dict: 統計情報
        """
        stats = {
            'total_blinks': self.blink_count,
            'current_blink_rate': self.get_blink_rate(),
            'avg_duration': np.mean(self.blink_durations) if self.blink_durations else 0,
            'current_ear': self.ear_values[-1] if self.ear_values else 0,
            'calibrated': self.ear_open_threshold is not None
        }
        
        return stats

    def draw_landmarks(self, frame, landmarks):
        """
        ランドマークを画像に描画（デバッグ用）
        
        Args:
            frame: BGR画像フレーム
            landmarks: MediaPipeの顔ランドマーク
            
        Returns:
            frame: ランドマークを描画したフレーム
        """
        h, w = frame.shape[:2]
        
        # 目のランドマークを描画
        for indices, color in [(self.LEFT_EYE_INDICES, (0, 255, 0)), 
                                (self.RIGHT_EYE_INDICES, (0, 0, 255))]:
            points = []
            for idx in indices:
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 2, color, -1)
            
            # 目の輪郭を線で結ぶ
            for i in range(len(points)):
                cv2.line(frame, points[i], points[(i+1) % len(points)], color, 1)
        
        return frame

    def __del__(self):
        """デストラクタ: MediaPipeリソースの解放"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()