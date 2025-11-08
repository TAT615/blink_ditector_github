"""
MediaPipe版瞬き検出器のテストスクリプト

使い方:
    python test_mediapipe_blink_detector.py

機能:
- リアルタイムで顔とランドマークを検出
- EAR値をリアルタイム表示
- 瞬き検出とカウント
- キャリブレーション機能
"""

import cv2
import numpy as np
import time
from blink_detector_mediapipe import BlinkDetectorMediaPipe


def main():
    """メイン関数"""
    print("=" * 60)
    print("MediaPipe版瞬き検出器 - テストプログラム")
    print("=" * 60)
    print()
    print("操作方法:")
    print("  [C] - キャリブレーション開始（5秒間）")
    print("  [R] - 統計情報をリセット")
    print("  [ESC] - 終了")
    print()
    print("=" * 60)
    
    # カメラ初期化
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ エラー: カメラを開けません")
        return
    
    # カメラ設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ カメラ初期化完了")
    
    # MediaPipe瞬き検出器の初期化
    detector = BlinkDetectorMediaPipe(
        buffer_size=300,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("✅ MediaPipe Face Mesh初期化完了")
    print()
    print("準備完了！[C]キーでキャリブレーションを開始してください")
    print()
    
    # FPS計測用
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("⚠️ フレーム取得失敗")
            break
        
        # フレームを左右反転（鏡像表示）
        frame = cv2.flip(frame, 1)
        
        # FPS計算
        fps_frame_count += 1
        if fps_frame_count >= 30:
            fps_end_time = time.time()
            fps = fps_frame_count / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
            fps_frame_count = 0
        
        # 顔とランドマークを検出
        landmarks = detector.detect_face_and_landmarks(frame)
        
        # ランドマークが検出された場合
        if landmarks is not None:
            # EAR計算
            current_ear = detector.calculate_ear_from_landmarks(landmarks, frame.shape)
            
            # キャリブレーション中の処理
            if detector.calibration_active:
                elapsed = time.time() - detector.calibration_start_time
                remaining = detector.calibration_duration - elapsed
                
                # 進捗バー
                progress = int((elapsed / detector.calibration_duration) * 100)
                bar_length = 30
                filled = int((progress / 100) * bar_length)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                # 画面に表示
                cv2.putText(frame, f"CALIBRATING: {remaining:.1f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"[{bar}] {progress}%", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 瞬き検出
            blink_info = detector.detect_blink(frame)
            
            # 瞬きが検出された場合
            if blink_info is not None:
                print(f"👁️ 瞬き検出! "
                      f"閉眼時間: {blink_info['closing_time']*1000:.1f}ms, "
                      f"開眼時間: {blink_info['opening_time']*1000:.1f}ms, "
                      f"係数: {blink_info['blink_coefficient']:.2f}")
            
            # ランドマークを描画（デバッグ用）
            frame = detector.draw_landmarks(frame, landmarks)
            
            # 統計情報を取得
            stats = detector.get_statistics()
            
            # 情報を画面に表示
            y_offset = 30
            line_height = 30
            
            # EAR値（色分け）
            ear_color = (0, 255, 0)  # 緑
            if detector.ear_closed_threshold and current_ear <= detector.ear_closed_threshold:
                ear_color = (0, 0, 255)  # 赤（閉眼）
            elif detector.ear_closing_threshold and current_ear <= detector.ear_closing_threshold:
                ear_color = (0, 165, 255)  # オレンジ（閉眼途中）
            
            cv2.putText(frame, f"EAR: {current_ear:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
            y_offset += line_height
            
            # 瞬き回数
            cv2.putText(frame, f"Blinks: {stats['total_blinks']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += line_height
            
            # 瞬き率
            cv2.putText(frame, f"Rate: {stats['current_blink_rate']:.1f}/min", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += line_height
            
            # 平均持続時間
            cv2.putText(frame, f"Avg Duration: {stats['avg_duration']*1000:.0f}ms", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += line_height
            
            # キャリブレーション状態
            calib_text = "Calibrated: YES" if stats['calibrated'] else "Calibrated: NO (Press C)"
            calib_color = (0, 255, 0) if stats['calibrated'] else (0, 0, 255)
            cv2.putText(frame, calib_text, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, calib_color, 2)
            y_offset += line_height
            
            # 瞬き状態
            state_names = {
                detector.BLINK_STATE_OPEN: "OPEN",
                detector.BLINK_STATE_CLOSING: "CLOSING",
                detector.BLINK_STATE_CLOSED: "CLOSED",
                detector.BLINK_STATE_OPENING: "OPENING"
            }
            state_text = f"State: {state_names.get(detector.blink_state, 'UNKNOWN')}"
            cv2.putText(frame, state_text, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
        else:
            # 顔が検出されない場合
            cv2.putText(frame, "No face detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # FPS表示
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # フレーム表示
        cv2.imshow("MediaPipe Blink Detector Test", frame)
        
        # キー入力処理
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n終了します...")
            break
        elif key == ord('c') or key == ord('C'):
            print("\n🎯 キャリブレーション開始")
            detector.start_calibration()
        elif key == ord('r') or key == ord('R'):
            print("\n🔄 統計情報をリセット")
            detector.blink_count = 0
            detector.blink_times.clear()
            detector.blink_durations.clear()
            detector.blink_details.clear()
    
    # 終了処理
    cap.release()
    cv2.destroyAllWindows()
    
    # 最終統計
    print("\n" + "=" * 60)
    print("最終統計")
    print("=" * 60)
    stats = detector.get_statistics()
    print(f"総瞬き回数: {stats['total_blinks']}")
    print(f"平均瞬き率: {stats['current_blink_rate']:.1f} 回/分")
    print(f"平均持続時間: {stats['avg_duration']*1000:.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
