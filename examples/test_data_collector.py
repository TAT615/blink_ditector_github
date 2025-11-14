"""
データ収集プログラムのテストスクリプト

プログラムが正しく動作するかをテストします。
"""

import sys
import os

def check_dependencies():
    """依存パッケージの確認"""
    print("=" * 60)
    print("依存パッケージの確認")
    print("=" * 60)
    
    packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy'
    }
    
    all_ok = True
    
    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
            print(f"✅ {package_name}: インストール済み")
        except ImportError:
            print(f"❌ {package_name}: 未インストール")
            print(f"   インストール: pip install {package_name} --break-system-packages")
            all_ok = False
    
    print()
    return all_ok


def check_camera():
    """カメラの確認"""
    print("=" * 60)
    print("カメラの確認")
    print("=" * 60)
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ カメラを開けませんでした")
            print("   カメラが接続されているか確認してください")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ フレームの取得に失敗しました")
            return False
        
        h, w = frame.shape[:2]
        print(f"✅ カメラ動作OK")
        print(f"   解像度: {w}x{h}")
        print()
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def check_mediapipe():
    """MediaPipeの確認"""
    print("=" * 60)
    print("MediaPipeの確認")
    print("=" * 60)
    
    try:
        import mediapipe as mp
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✅ MediaPipe Face Mesh: 正常に初期化")
        print()
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def check_output_directory():
    """出力ディレクトリの確認"""
    print("=" * 60)
    print("出力ディレクトリの確認")
    print("=" * 60)
    
    output_dir = "data/sessions"
    
    if os.path.exists(output_dir):
        print(f"✅ ディレクトリ存在: {output_dir}")
        
        # JSONファイルの数を確認
        json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        print(f"   既存のJSONファイル: {len(json_files)}個")
    else:
        print(f"ℹ️  ディレクトリ未作成: {output_dir}")
        print(f"   プログラム実行時に自動作成されます")
    
    print()
    return True


def test_program():
    """プログラムファイルの確認"""
    print("=" * 60)
    print("プログラムファイルの確認")
    print("=" * 60)
    
    program_file = "src/drowsiness_data_collector_with_ellipse.py"
    
    if os.path.exists(program_file):
        print(f"✅ プログラムファイル存在: {program_file}")
        
        # ファイルサイズ
        size = os.path.getsize(program_file)
        print(f"   ファイルサイズ: {size:,} bytes")
    else:
        print(f"❌ プログラムファイルが見つかりません: {program_file}")
        print(f"   ダウンロードしたファイルを確認してください")
        return False
    
    print()
    return True


def print_usage():
    """使い方を表示"""
    print("=" * 60)
    print("使い方")
    print("=" * 60)
    print()
    print("【正常状態データ収集】")
    print("  python drowsiness_data_collector_with_ellipse.py --label 0")
    print()
    print("【眠気状態データ収集】")
    print("  python drowsiness_data_collector_with_ellipse.py --label 1")
    print()
    print("【操作方法】")
    print("  ESC または Q: 終了・保存")
    print()
    print("=" * 60)


def main():
    """メイン関数"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "データ収集プログラム - 動作確認" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    results = []
    
    # 依存パッケージの確認
    results.append(("依存パッケージ", check_dependencies()))
    
    # カメラの確認
    results.append(("カメラ", check_camera()))
    
    # MediaPipeの確認
    results.append(("MediaPipe", check_mediapipe()))
    
    # 出力ディレクトリの確認
    results.append(("出力ディレクトリ", check_output_directory()))
    
    # プログラムファイルの確認
    results.append(("プログラムファイル", test_program()))
    
    # 結果サマリー
    print("=" * 60)
    print("確認結果サマリー")
    print("=" * 60)
    
    all_ok = True
    for name, result in results:
        status = "✅ OK" if result else "❌ NG"
        print(f"{status}: {name}")
        if not result:
            all_ok = False
    
    print("=" * 60)
    print()
    
    if all_ok:
        print("🎉 すべてのチェックが完了しました！")
        print("   プログラムを実行できます。")
        print()
        print_usage()
    else:
        print("⚠️  いくつかの問題があります。")
        print("   上記のエラーメッセージを確認して、修正してください。")
        print()
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
