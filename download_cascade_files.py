"""
OpenCV Haar Cascade ファイルダウンロードスクリプト

このスクリプトは、OpenCVのHaar Cascadeファイルをダウンロードします。
Windowsの日本語パスでOpenCVのファイルが見つからない場合に使用してください。
"""

import os
import urllib.request
import sys

def download_file(url, filename):
    """
    ファイルをダウンロードする
    
    Args:
        url (str): ダウンロードURL
        filename (str): 保存ファイル名
    """
    try:
        print(f"📥 ダウンロード中: {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✅ 成功: {filename}")
        return True
    except Exception as e:
        print(f"❌ エラー: {filename} - {e}")
        return False

def main():
    """メイン処理"""
    print("=" * 70)
    print("OpenCV Haar Cascade ファイル ダウンロードツール")
    print("=" * 70)
    
    # ダウンロードするファイルのリスト
    files = {
        'haarcascade_frontalface_default.xml': 
            'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
        'haarcascade_eye.xml':
            'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml'
    }
    
    # カレントディレクトリを確認
    current_dir = os.getcwd()
    print(f"\n📁 保存先: {current_dir}\n")
    
    # 各ファイルをダウンロード
    success_count = 0
    for filename, url in files.items():
        # ファイルが既に存在するか確認
        if os.path.exists(filename):
            print(f"⚠️ スキップ: {filename} (既に存在します)")
            success_count += 1
        else:
            if download_file(url, filename):
                success_count += 1
    
    print("\n" + "=" * 70)
    if success_count == len(files):
        print("✅ すべてのファイルのダウンロードが完了しました！")
        print("\nこれで眠気検知システムを実行できます:")
        print("  python -m src.drowsiness_data_collector")
    else:
        print(f"⚠️ {len(files) - success_count} 個のファイルのダウンロードに失敗しました")
        print("\n手動でダウンロードする場合:")
        for filename, url in files.items():
            if not os.path.exists(filename):
                print(f"\n  {filename}:")
                print(f"  {url}")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        sys.exit(1)
