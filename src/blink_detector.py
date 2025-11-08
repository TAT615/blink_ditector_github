"""
瞬き検出モジュール（OpenCV Haar Cascade版）
OpenCV Haar Cascadeを使用して顔検出し、目のランドマークから瞬きを検出します。
"""

import os
import numpy as np
import cv2
import time
from collections import deque


class BlinkDetector:
    """
    OpenCV Haar Cascadeを使用した瞬き検出器
    
    特徴:
    - 顔検出: OpenCV Haar Cascade
    - 目検出: 目のHaar Cascade
    - EAR (Eye Aspect Ratio) による瞬き検出
    - 個人キャリブレーション機能
    """
    
    # 瞬き状態の定義（4段階）
    BLINK_STATE_OPEN = 0          # 完全開眼
    BLINK_STATE_CLOSING = 1       # 閉眼途中
    BLINK_STATE_CLOSED = 2        # 完全閉眼
    BLINK_STATE_OPENING = 3       # 開眼途中
    
    def __init__(self, buffer_size=300):
        """
        初期化
        
        Args:
            buffer_size (int): データバッファサイズ
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
        
        # OpenCV Haar Cascade分類器の初期化
        self._initialize_cascades()
        
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
        
        # 目の検出結果を保存（視覚化用）
        self.detected_eyes = []  # 検出された目の矩形リスト
        self.left_eye_rect = None  # 左目の矩形
        self.right_eye_rect = None  # 右目の矩形
        self.face_rect_cache = None  # 顔の矩形（目の座標を画面座標に変換するため）
        
        # キャリブレーション状態の記録
        self.calibration_log = []
        
        # 面積ベース検出用の追加属性
        self.area_state = 'unknown'
        self.prev_area_state = 'unknown'
        self.area_thresholds_cache = None
        self.last_threshold_update = 0

    def _initialize_cascades(self):
        """
        Haar Cascade分類器を初期化
        複数の方法でファイルを探し、エラーハンドリングを行う
        """
        # 方法1: cv2.data.haarcascades を使用
        try:
            cascade_path = cv2.data.haarcascades
            face_cascade_file = os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
            eye_cascade_file = os.path.join(cascade_path, 'haarcascade_eye.xml')
            
            # ファイルの存在確認
            if os.path.exists(face_cascade_file) and os.path.exists(eye_cascade_file):
                self.face_cascade = cv2.CascadeClassifier(face_cascade_file)
                self.eye_cascade = cv2.CascadeClassifier(eye_cascade_file)
                
                # 正しく読み込まれたか確認
                if not self.face_cascade.empty() and not self.eye_cascade.empty():
                    print(f"✅ Haar Cascade ファイル読み込み成功 (cv2.data)")
                    return
        except Exception:
            pass  # 静かに次の方法を試す
        
        # 方法2: カレントディレクトリに配置されたファイルを使用
        try:
            face_cascade_file = 'haarcascade_frontalface_default.xml'
            eye_cascade_file = 'haarcascade_eye.xml'
            
            if os.path.exists(face_cascade_file) and os.path.exists(eye_cascade_file):
                self.face_cascade = cv2.CascadeClassifier(face_cascade_file)
                self.eye_cascade = cv2.CascadeClassifier(eye_cascade_file)
                
                if not self.face_cascade.empty() and not self.eye_cascade.empty():
                    print(f"✅ Haar Cascade ファイル読み込み成功 (カレントディレクトリ)")
                    return
        except Exception:
            pass  # 静かに次の方法を試す
        
        # 方法3: データディレクトリを使用
        try:
            data_dir = 'data'
            if os.path.exists(data_dir):
                face_cascade_file = os.path.join(data_dir, 'haarcascade_frontalface_default.xml')
                eye_cascade_file = os.path.join(data_dir, 'haarcascade_eye.xml')
                
                if os.path.exists(face_cascade_file) and os.path.exists(eye_cascade_file):
                    self.face_cascade = cv2.CascadeClassifier(face_cascade_file)
                    self.eye_cascade = cv2.CascadeClassifier(eye_cascade_file)
                    
                    if not self.face_cascade.empty() and not self.eye_cascade.empty():
                        print(f"✅ Haar Cascade ファイル読み込み成功 (dataディレクトリ)")
                        return
        except Exception:
            pass  # 静かに失敗する
        
        # すべての方法が失敗した場合
        print("\n" + "="*70)
        print("❌ Haar Cascade ファイルが見つかりませんでした")
        print("="*70)
        print("\n解決方法:")
        print("1. 以下のコマンドでファイルをダウンロードしてください:")
        print("   curl -O https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
        print("   curl -O https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml")
        print("\n2. ダウンロードしたファイルをプロジェクトのルートディレクトリに配置してください")
        print("\n3. または、以下のコマンドでOpenCVを再インストールしてください:")
        print("   pip uninstall opencv-python")
        print("   pip install opencv-python")
        print("="*70)
        
        raise RuntimeError("Haar Cascade ファイルが見つかりません。上記の解決方法を試してください。")

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

    def detect_face(self, frame):
        """
        顔を検出する（改善版）
        
        Args:
            frame: BGR画像フレーム
            
        Returns:
            tuple: (x, y, w, h) または None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ヒストグラム平坦化で照明変化に対応
        gray = cv2.equalizeHist(gray)
        
        # 顔検出（複数のスケールで試行）
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(100, 100),  # より大きな最小サイズ
            maxSize=(400, 400),  # 最大サイズも制限
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            # 最も大きな顔を選択（最も近い顔と仮定）
            face = max(faces, key=lambda f: f[2] * f[3])
            self.face_lost_frames = 0  # 検出成功時はリセット
            return tuple(face)
        
        # 顔が検出できない場合
        self.face_lost_frames += 1
        
        # 一定フレーム数検出できない場合は状態をリセット
        if self.face_lost_frames > self.MAX_FACE_LOST_FRAMES * 2:
            self._reset_detection_state()
        
        return None
    
    def _reset_detection_state(self):
        """検出状態をリセット"""
        self.face_lost_frames = 0
        self.last_valid_ear = None
        # 瞬き状態は継続（キャリブレーション結果は保持）

    def calculate_ear_from_eyes(self, frame, face_rect):
        """
        目の領域からEARを計算（改善版）
        
        Args:
            frame: BGR画像フレーム
            face_rect: 顔の矩形 (x, y, w, h)
            
        Returns:
            float: EAR値（0.0-1.0）
        """
        x, y, w, h = face_rect
        
        # 顔の矩形をキャッシュ（描画用）
        self.face_rect_cache = face_rect
        
        # 顔領域を抽出
        face_roi = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # 目を検出（検出しやすいパラメータに調整）
        eyes = self.eye_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.1,   # スケールを少し粗く（検出しやすく）
            minNeighbors=5,    # 検出条件を緩く（より多く検出）
            minSize=(int(w*0.08), int(h*0.08)),  # 最小サイズを少し小さく
            maxSize=(int(w*0.5), int(h*0.4))     # 最大サイズを少し大きく
        )
        
        # 検出された目を保存（視覚化用）
        self.detected_eyes = eyes
        self.left_eye_rect = None
        self.right_eye_rect = None
        
        if len(eyes) >= 2:
            # 2つ以上の目が検出された場合
            # Y座標が近い（同じ高さにある）2つの目を選択
            eyes_sorted = sorted(eyes, key=lambda e: e[0])  # X座標でソート
            
            # 左右の目を判定（X座標で判定）
            left_eye = eyes_sorted[0]
            right_eye = eyes_sorted[1] if len(eyes_sorted) > 1 else eyes_sorted[0]
            
            # 目の高さが大きく異なる場合は信頼性が低い
            if abs(left_eye[1] - right_eye[1]) > h * 0.2:
                # 高さが近い別のペアを探す
                for i in range(len(eyes_sorted) - 1):
                    for j in range(i + 1, len(eyes_sorted)):
                        if abs(eyes_sorted[i][1] - eyes_sorted[j][1]) < h * 0.1:
                            left_eye = eyes_sorted[i]
                            right_eye = eyes_sorted[j]
                            break
            
            # 左右の目を保存（視覚化用）
            self.left_eye_rect = left_eye
            self.right_eye_rect = right_eye
            
            # 各目のEARを計算
            left_ear = self._estimate_ear_from_eye_rect(left_eye)
            right_ear = self._estimate_ear_from_eye_rect(right_eye)
            
            # 左右の目のEAR値を記録
            self.left_eye_areas.append(left_eye[2] * left_eye[3])
            self.right_eye_areas.append(right_eye[2] * right_eye[3])
            self.total_eye_areas.append(left_eye[2] * left_eye[3] + right_eye[2] * right_eye[3])
            
            # 平均EAR
            avg_ear = (left_ear + right_ear) / 2.0
            
            # スムージング（前回の値との加重平均）
            if self.last_valid_ear is not None:
                avg_ear = 0.7 * avg_ear + 0.3 * self.last_valid_ear
            
            return avg_ear
            
        elif len(eyes) == 1:
            # 1つの目のみ検出（信頼性低）
            self.left_eye_rect = eyes[0]  # 視覚化用に保存
            eye_ear = self._estimate_ear_from_eye_rect(eyes[0])
            
            # 前回の値とブレンド（ノイズ削減）
            if self.last_valid_ear is not None:
                eye_ear = 0.5 * eye_ear + 0.5 * self.last_valid_ear
            
            return eye_ear
            
        else:
            # 目が検出できない場合
            self.face_lost_frames += 1
            
            # 少しの間は前回の値を使用
            if self.last_valid_ear is not None and self.face_lost_frames < self.MAX_FACE_LOST_FRAMES:
                return self.last_valid_ear
            
            # それでもダメな場合はデフォルト値
            return 0.3

    def _estimate_ear_from_eye_rect(self, eye_rect):
        """
        目の矩形からEARを推定（改善版）
        
        Args:
            eye_rect: 目の矩形 (x, y, w, h)
            
        Returns:
            float: 推定EAR値
        """
        ex, ey, ew, eh = eye_rect
        
        if ew == 0:
            return 0.3
        
        # 高さと幅の比率からEARを推定
        # 目が開いているとき: 横長（高さ/幅が小さい）→ EARが大きい
        # 目が閉じているとき: 縦長（高さ/幅が大きい）→ EARが小さい
        aspect_ratio = eh / ew
        
        # より正確なマッピング
        # aspect_ratio: 0.3-0.7の範囲を想定
        # EAR: 0.15-0.40の範囲にマッピング
        if aspect_ratio < 0.35:
            # 目が非常に開いている
            ear = 0.35 + (0.35 - aspect_ratio) * 0.5
        elif aspect_ratio > 0.6:
            # 目が閉じている
            ear = 0.15 + (0.7 - aspect_ratio) * 0.2
        else:
            # 通常の開眼状態
            ear = 0.15 + (0.7 - aspect_ratio) * 0.4
        
        return max(0.10, min(0.50, ear))

    def detect_blink(self, frame, face_rect):
        """
        瞬きを検出
        
        Args:
            frame: BGR画像フレーム
            face_rect: 顔の矩形 (x, y, w, h)
            
        Returns:
            tuple: (blink_detected, ear_value, blink_state)
        """
        # EAR値を計算
        ear = self.calculate_ear_from_eyes(frame, face_rect)
        self.last_valid_ear = ear
        self.ear_values.append(ear)
        
        # キャリブレーション中の場合
        if self.calibration_active:
            calibration_done = self.update_calibration(ear)
            return False, ear, self.BLINK_STATE_OPEN
        
        # 閾値が設定されていない場合はデフォルトを使用
        if self.ear_open_threshold is None:
            self._use_default_thresholds()
        
        # 瞬き検出のステートマシン
        blink_detected = False
        
        if self.blink_state == self.BLINK_STATE_OPEN:
            if ear < self.ear_closing_threshold:
                self.blink_state = self.BLINK_STATE_CLOSING
                self.current_blink_start = time.time()
                self.current_blink_min_ear = ear
                
        elif self.blink_state == self.BLINK_STATE_CLOSING:
            if ear < self.current_blink_min_ear:
                self.current_blink_min_ear = ear
            if ear < self.ear_closed_threshold:
                self.blink_state = self.BLINK_STATE_CLOSED
                self.current_blink_closed = time.time()
                
        elif self.blink_state == self.BLINK_STATE_CLOSED:
            if ear > self.ear_closed_threshold:
                self.blink_state = self.BLINK_STATE_OPENING
                
        elif self.blink_state == self.BLINK_STATE_OPENING:
            if ear > self.ear_opening_threshold:
                # 瞬き完了
                blink_detected = True
                self.blink_count += 1
                current_time = time.time()
                self.blink_times.append(current_time)
                self.recent_blinks.append(current_time)
                
                # 瞬き時間を記録
                if self.current_blink_start is not None and self.current_blink_closed is not None:
                    # t1: 閉眼時間（開眼開始→完全閉眼）
                    t1 = self.current_blink_closed - self.current_blink_start
                    # t2: 閉眼持続時間（完全閉眼の時間）
                    t2_end = current_time  # 開眼完了時刻
                    t2 = t2_end - self.current_blink_closed
                    # t3: 開眼時間（完全閉眼→開眼完了）
                    t3 = current_time - self.current_blink_closed
                    # 全体の瞬き時間
                    duration = current_time - self.current_blink_start
                    self.blink_durations.append(duration)
                    
                    # 瞬き詳細情報を記録
                    blink_detail = {
                        't1': t1,  # 閉眼時間
                        't2': t2,  # 閉眼持続時間
                        't3': t3,  # 開眼時間
                        'ear_min': self.current_blink_min_ear,  # 最小EAR値
                        'total_duration': duration,  # 全体の時間
                        'timestamp': current_time
                    }
                    self.blink_details.append(blink_detail)
                
                self.blink_state = self.BLINK_STATE_OPEN
                self.current_blink_start = None
                self.current_blink_closed = None
                self.current_blink_min_ear = 1.0
        
        return blink_detected, ear, self.blink_state

    def get_blink_rate(self, window_seconds=60):
        """
        瞬き率を計算
        
        Args:
            window_seconds (int): 計算ウィンドウ（秒）
            
        Returns:
            float: 1分あたりの瞬き回数
        """
        current_time = time.time()
        recent = [t for t in self.recent_blinks if current_time - t < window_seconds]
        
        if len(recent) < 2:
            return 0.0
        
        return len(recent) * (60.0 / window_seconds)

    def get_statistics(self):
        """
        統計情報を取得（改善版）
        
        Returns:
            dict: 統計情報
        """
        stats = {
            'blink_count': self.blink_count,
            'blink_rate': self.get_blink_rate(),
            'avg_ear': np.mean(list(self.ear_values)) if self.ear_values else 0.0,
            'current_ear': self.last_valid_ear if self.last_valid_ear else 0.0,
            'calibrated': not self.calibration_active and self.ear_open_threshold is not None,
            'blink_state': self.blink_state,
            'face_lost_frames': self.face_lost_frames
        }
        
        # 瞬き時間の統計
        if self.blink_durations:
            stats['avg_blink_duration'] = np.mean(list(self.blink_durations))
            stats['min_blink_duration'] = np.min(list(self.blink_durations))
            stats['max_blink_duration'] = np.max(list(self.blink_durations))
        
        # 目の面積統計
        if self.total_eye_areas:
            stats['avg_eye_area'] = np.mean(list(self.total_eye_areas))
            stats['current_eye_area'] = self.total_eye_areas[-1] if self.total_eye_areas else 0
        
        # キャリブレーション情報
        if self.ear_open_threshold is not None:
            stats['ear_open_threshold'] = self.ear_open_threshold
            stats['ear_closed_threshold'] = self.ear_closed_threshold
            stats['ear_dynamic_range'] = self.ear_dynamic_range if self.ear_dynamic_range else 0
        
        return stats
    
    def draw_debug_info(self, frame, face_rect=None):
        """
        デバッグ情報を画面に表示
        
        Args:
            frame: BGR画像フレーム
            face_rect: 顔の矩形 (x, y, w, h) または None
        """
        try:
            stats = self.get_statistics()
            
            # 顔の矩形を描画
            if face_rect is not None:
                x, y, w, h = face_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 検出された目を描画
                self._draw_detected_eyes(frame, face_rect)
            
            # 左側の情報
            y_offset = 30
            line_height = 30
            
            # 基本情報
            cv2.putText(frame, f"Blinks: {stats['blink_count']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += line_height
            
            cv2.putText(frame, f"Rate: {stats['blink_rate']:.1f}/min", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += line_height
            
            # EAR情報
            ear_color = (0, 255, 0)
            if stats['current_ear'] < 0.25:
                ear_color = (0, 0, 255)  # 赤（閉眼）
            elif stats['current_ear'] < 0.30:
                ear_color = (0, 255, 255)  # 黄（中間）
            
            cv2.putText(frame, f"EAR: {stats['current_ear']:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
            y_offset += line_height
            
            # 状態表示
            state_names = {0: "OPEN", 1: "CLOSING", 2: "CLOSED", 3: "OPENING"}
            state_colors = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255), 3: (255, 0, 255)}
            state_name = state_names.get(stats['blink_state'], "UNKNOWN")
            state_color = state_colors.get(stats['blink_state'], (255, 255, 255))
            
            cv2.putText(frame, f"State: {state_name}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            y_offset += line_height
            
            # キャリブレーション状態
            if stats['calibrated']:
                cv2.putText(frame, "Calibrated", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not Calibrated", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += line_height
            
            # 右側の閾値情報
            if stats['calibrated'] and 'ear_open_threshold' in stats:
                right_x = frame.shape[1] - 250
                y_offset = 30
                
                cv2.putText(frame, "Thresholds:", 
                           (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += line_height
                
                cv2.putText(frame, f"Open: {stats['ear_open_threshold']:.3f}", 
                           (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                y_offset += 25
                
                if self.ear_opening_threshold:
                    cv2.putText(frame, f"Opening: {self.ear_opening_threshold:.3f}", 
                               (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
                    y_offset += 25
                
                if self.ear_closing_threshold:
                    cv2.putText(frame, f"Closing: {self.ear_closing_threshold:.3f}", 
                               (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    y_offset += 25
                
                cv2.putText(frame, f"Closed: {stats['ear_closed_threshold']:.3f}", 
                           (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            # キャリブレーション中の表示
            if self.calibration_active:
                elapsed = time.time() - self.calibration_start_time
                remaining = max(0, self.calibration_duration - elapsed)
                
                cv2.putText(frame, f"CALIBRATING: {remaining:.1f}s", 
                           (frame.shape[1]//2 - 150, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                
                cv2.putText(frame, "Please blink naturally", 
                           (frame.shape[1]//2 - 150, frame.shape[0]//2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        except Exception as e:
            print(f"Error in draw_debug_info: {e}")
    
    def _draw_detected_eyes(self, frame, face_rect):
        """
        検出された目を描画
        
        Args:
            frame: BGR画像フレーム
            face_rect: 顔の矩形 (x, y, w, h)
        """
        if face_rect is None or self.face_rect_cache is None:
            return
        
        face_x, face_y, face_w, face_h = face_rect
        
        # すべての検出された目を薄い色で描画（参考用）
        for ex, ey, ew, eh in self.detected_eyes:
            # 顔座標系から画面座標系に変換
            abs_x = face_x + ex
            abs_y = face_y + ey
            
            # 目の矩形を描画（薄い青）
            cv2.rectangle(frame, (abs_x, abs_y), (abs_x + ew, abs_y + eh), 
                         (200, 200, 100), 1)
        
        # 左目を描画（使用している目）
        if self.left_eye_rect is not None:
            ex, ey, ew, eh = self.left_eye_rect
            abs_x = face_x + ex
            abs_y = face_y + ey
            
            # 左目の矩形（青）
            cv2.rectangle(frame, (abs_x, abs_y), (abs_x + ew, abs_y + eh), 
                         (255, 0, 0), 2)
            
            # 左目の中心点（青い円）
            center_x = abs_x + ew // 2
            center_y = abs_y + eh // 2
            cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
            
            # "L"ラベル
            cv2.putText(frame, "L", (abs_x, abs_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 右目を描画（使用している目）
        if self.right_eye_rect is not None:
            ex, ey, ew, eh = self.right_eye_rect
            abs_x = face_x + ex
            abs_y = face_y + ey
            
            # 右目の矩形（赤）
            cv2.rectangle(frame, (abs_x, abs_y), (abs_x + ew, abs_y + eh), 
                         (0, 0, 255), 2)
            
            # 右目の中心点（赤い円）
            center_x = abs_x + ew // 2
            center_y = abs_y + eh // 2
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            
            # "R"ラベル
            cv2.putText(frame, "R", (abs_x, abs_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 目の検出状態を表示
        eye_status_y = face_y + face_h + 20
        if len(self.detected_eyes) == 0:
            cv2.putText(frame, "Eyes: NOT DETECTED", 
                       (face_x, eye_status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif len(self.detected_eyes) == 1:
            cv2.putText(frame, "Eyes: 1 detected", 
                       (face_x, eye_status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f"Eyes: {len(self.detected_eyes)} detected", 
                       (face_x, eye_status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)