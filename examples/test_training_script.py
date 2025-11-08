"""
訓練スクリプトの動作確認テスト
簡単なダミーデータで訓練パイプラインをテスト
"""

import numpy as np
import os
import sys

print("=" * 70)
print("🧪 訓練スクリプト動作確認テスト")
print("=" * 70)

# ダミーデータの作成
print("\n📦 ダミーデータ作成中...")

# テスト用ディレクトリ
test_data_dir = "test_drowsiness_data"
sessions_dir = os.path.join(test_data_dir, 'sessions')
sequences_dir = os.path.join(test_data_dir, 'sequences')

os.makedirs(sessions_dir, exist_ok=True)
os.makedirs(sequences_dir, exist_ok=True)

# ダミーシーケンスデータ生成
np.random.seed(42)

# 正常状態のデータ
normal_sequences = np.random.randn(50, 10, 6).astype(np.float32)
normal_sequences[:, :, 0] = np.abs(normal_sequences[:, :, 0]) + 1.2  # 瞬き係数高め
normal_labels = np.zeros(50, dtype=np.int64)

# 眠気状態のデータ
drowsy_sequences = np.random.randn(50, 10, 6).astype(np.float32)
drowsy_sequences[:, :, 0] = np.abs(drowsy_sequences[:, :, 0]) + 0.6  # 瞬き係数低め
drowsy_sequences[:, :, 1:3] = np.abs(drowsy_sequences[:, :, 1:3]) + 0.5  # 時間長め
drowsy_labels = np.ones(50, dtype=np.int64)

# データ保存
np.savez(os.path.join(sequences_dir, 'normal_test_sequences.npz'),
         sequences=normal_sequences,
         labels=normal_labels,
         session_name='normal_test')

np.savez(os.path.join(sequences_dir, 'drowsy_test_sequences.npz'),
         sequences=drowsy_sequences,
         labels=drowsy_labels,
         session_name='drowsy_test')

print(f"✅ ダミーデータ作成完了")
print(f"   正常: {len(normal_sequences)} シーケンス")
print(f"   眠気: {len(drowsy_sequences)} シーケンス")

# 訓練スクリプトのインポートとテスト
try:
    from src.train_drowsiness_model import ModelTrainer, create_default_config
    print("\n✅ src.train_drowsiness_model のインポート成功")
except ImportError as e:
    print(f"\n❌ インポートエラー: {e}")
    print("   src/train_drowsiness_model.py が必要です")
    print("   プロジェクトルートから実行してください: python examples/test_training_script.py")
    sys.exit(1)

# テスト設定
print("\n🔧 テスト設定作成...")
config = create_default_config()
config['epochs'] = 10  # テスト用に短く
config['batch_size'] = 16
config['output_dir'] = 'test_models'
config['show_plots'] = False

print(f"   エポック数: {config['epochs']}")
print(f"   バッチサイズ: {config['batch_size']}")

# トレーナー作成
print("\n🎓 トレーナー作成...")
trainer = ModelTrainer(config)

# 訓練パイプライン実行
print("\n🚀 訓練パイプライン実行...")
success = trainer.run_full_training_pipeline(test_data_dir)

if success:
    print("\n" + "=" * 70)
    print("✅ テスト成功！")
    print("=" * 70)
    print("訓練スクリプトは正常に動作しています。")
    print("実際のデータで訓練する準備ができました。")
    print("\n次のコマンドで実際の訓練を開始できます:")
    print("  python train_drowsiness_model.py --data-dir drowsiness_training_data")
    print("=" * 70)
else:
    print("\n" + "=" * 70)
    print("❌ テスト失敗")
    print("=" * 70)
    print("エラーを確認してください。")
    print("=" * 70)

# クリーンアップ（オプション）
print("\n🧹 テストデータをクリーンアップしますか? (y/n): ", end='')
try:
    response = input().strip().lower()
    if response == 'y':
        import shutil
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
        if os.path.exists('test_models'):
            shutil.rmtree('test_models')
        print("✅ クリーンアップ完了")
except:
    print("\n⚠️ クリーンアップをスキップ")
