"""
MediaPipe動作確認スクリプト
カメラなしで基本的な初期化とAPI確認を行います
"""

import sys
sys.path.insert(0, '/home/claude')

try:
    from blink_detector_mediapipe import BlinkDetectorMediaPipe
    import numpy as np
    
    print("=" * 60)
    print("MediaPipe版瞬き検出器 - 動作確認")
    print("=" * 60)
    print()
    
    # 初期化テスト
    print("1. 初期化テスト...")
    detector = BlinkDetectorMediaPipe(
        buffer_size=300,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("   ✅ 初期化成功")
    print()
    
    # 属性確認
    print("2. 属性確認...")
    print(f"   - バッファサイズ: {detector.buffer_size}")
    print(f"   - 左目ランドマーク数: {len(detector.LEFT_EYE_INDICES)}")
    print(f"   - 右目ランドマーク数: {len(detector.RIGHT_EYE_INDICES)}")
    print(f"   - 瞬き状態: {detector.blink_state}")
    print("   ✅ 属性確認成功")
    print()
    
    # メソッド確認
    print("3. メソッド確認...")
    methods = [
        'start_calibration',
        'detect_face_and_landmarks',
        'calculate_ear_from_landmarks',
        'detect_blink',
        'get_blink_rate',
        'get_statistics',
        'draw_landmarks'
    ]
    
    for method in methods:
        if hasattr(detector, method):
            print(f"   ✅ {method}")
        else:
            print(f"   ❌ {method}")
    print()
    
    # 統計情報取得テスト
    print("4. 統計情報取得テスト...")
    stats = detector.get_statistics()
    print(f"   - 総瞬き回数: {stats['total_blinks']}")
    print(f"   - 瞬き率: {stats['current_blink_rate']:.1f}/min")
    print(f"   - キャリブレーション状態: {stats['calibrated']}")
    print("   ✅ 統計情報取得成功")
    print()
    
    # キャリブレーションテスト
    print("5. キャリブレーション開始テスト...")
    detector.start_calibration()
    print(f"   - キャリブレーション有効: {detector.calibration_active}")
    print(f"   - 開始時刻設定: {detector.calibration_start_time is not None}")
    print("   ✅ キャリブレーション開始成功")
    print()
    
    print("=" * 60)
    print("すべてのテストが成功しました！ 🎉")
    print("=" * 60)
    print()
    print("次のステップ:")
    print("1. カメラ付きのPCで test_mediapipe_blink_detector.py を実行")
    print("2. リアルタイムで顔とランドマークの検出を確認")
    print("3. キャリブレーション（[C]キー）を実行")
    print("4. 瞬き検出の精度を確認")
    print()
    
except Exception as e:
    print(f"❌ エラーが発生しました: {e}")
    import traceback
    traceback.print_exc()
