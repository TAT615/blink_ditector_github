"""
訓練済みモデルのロードと使用サンプル
Load and Use Trained Model Sample

訓練済みモデルを読み込んで予測する方法を示します。
"""

import numpy as np
import json
import os
import sys

# 自作モジュールのインポート
try:
    from src.lstm_drowsiness_model import DrowsinessEstimator
    from src.blink_feature_extractor import BlinkFeatureExtractor
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    print("   プロジェクトルートから実行してください: python examples/load_and_use_model.py")
    sys.exit(1)


def load_trained_model(model_path: str) -> DrowsinessEstimator:
    """
    訓練済みモデルを読み込み
    
    Args:
        model_path (str): モデルファイルパス (.pth)
        
    Returns:
        DrowsinessEstimator: 読み込まれたモデル
    """
    print("=" * 70)
    print("📂 訓練済みモデルの読み込み")
    print("=" * 70)
    
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return None
    
    # 推定器作成
    estimator = DrowsinessEstimator()
    
    # モデル読み込み
    estimator.load_model(model_path)
    
    print(f"✅ モデル読み込み完了: {model_path}")
    
    # メタデータがあれば表示
    metadata_path = model_path.replace('.pth', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("\n📊 モデル情報:")
        print(f"   モデル名: {metadata.get('model_name', 'N/A')}")
        print(f"   訓練日時: {metadata.get('timestamp', 'N/A')}")
        print(f"   テスト精度: {metadata.get('test_accuracy', 'N/A'):.2f}%")
        
        if 'data_statistics' in metadata:
            stats = metadata['data_statistics']
            print(f"   訓練データ数: {stats.get('train_count', 'N/A')}")
    
    print("=" * 70)
    
    return estimator


def load_normalization_params(params_path: str) -> dict:
    """
    正規化パラメータを読み込み
    
    Args:
        params_path (str): 正規化パラメータファイルパス
        
    Returns:
        dict: 正規化パラメータ
    """
    if not os.path.exists(params_path):
        print(f"⚠️ 正規化パラメータが見つかりません: {params_path}")
        return None
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # リストをNumPy配列に変換
    for key, value in params.items():
        if isinstance(value, list):
            params[key] = np.array(value, dtype=np.float32)
    
    print(f"✅ 正規化パラメータ読み込み: {params_path}")
    return params


def predict_drowsiness(estimator: DrowsinessEstimator, 
                      sequence: np.ndarray,
                      normalization_params: dict = None) -> dict:
    """
    眠気状態を予測
    
    Args:
        estimator: 訓練済みモデル
        sequence: 入力シーケンス (1, 10, 6) or (10, 6)
        normalization_params: 正規化パラメータ
        
    Returns:
        dict: 予測結果
    """
    # シーケンスの形状を確認
    if sequence.ndim == 2:
        sequence = sequence[np.newaxis, ...]  # (10, 6) -> (1, 10, 6)
    
    # 正規化
    if normalization_params is not None and normalization_params.get('is_fitted', False):
        mean = normalization_params['mean']
        std = normalization_params['std']
        
        original_shape = sequence.shape
        sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])
        sequence_normalized = (sequence_reshaped - mean) / std
        sequence = sequence_normalized.reshape(original_shape)
    
    # 予測
    pred_class = estimator.predict(sequence)[0]
    pred_proba = estimator.predict_proba(sequence)[0]
    
    # 結果
    result = {
        'class': int(pred_class),
        'class_name': '正常' if pred_class == 0 else '眠気',
        'normal_probability': float(pred_proba[0]),
        'drowsy_probability': float(pred_proba[1]),
        'confidence': float(max(pred_proba))
    }
    
    return result


def demo_prediction():
    """
    予測のデモンストレーション
    """
    print("\n" + "=" * 70)
    print("🔮 予測デモンストレーション")
    print("=" * 70)
    
    # モデルパスの入力
    print("\n訓練済みモデルのパスを入力してください:")
    print("（例: trained_models/drowsiness_lstm_20240101_120000.pth）")
    model_path = input("モデルパス: ").strip()
    
    if not model_path:
        print("⚠️ デモ用にダミーデータで実行します")
        return demo_with_dummy_data()
    
    # モデル読み込み
    estimator = load_trained_model(model_path)
    if estimator is None:
        return
    
    # 正規化パラメータ読み込み
    norm_params_path = "drowsiness_training_data/normalization_params.json"
    normalization_params = load_normalization_params(norm_params_path)
    
    # サンプルシーケンスの生成（実際にはリアルタイムデータから取得）
    print("\n📊 サンプルシーケンスで予測を実行...")
    
    # 正常状態のサンプル
    normal_sample = np.random.randn(10, 6).astype(np.float32)
    normal_sample[:, 0] = np.abs(normal_sample[:, 0]) + 1.2  # 瞬き係数高め
    
    result = predict_drowsiness(estimator, normal_sample, normalization_params)
    
    print("\n🔍 正常状態サンプルの予測結果:")
    print(f"   予測: {result['class_name']}")
    print(f"   正常確率: {result['normal_probability']:.1%}")
    print(f"   眠気確率: {result['drowsy_probability']:.1%}")
    print(f"   信頼度: {result['confidence']:.1%}")
    
    # 眠気状態のサンプル
    drowsy_sample = np.random.randn(10, 6).astype(np.float32)
    drowsy_sample[:, 0] = np.abs(drowsy_sample[:, 0]) + 0.6  # 瞬き係数低め
    drowsy_sample[:, 1:3] = np.abs(drowsy_sample[:, 1:3]) + 0.5  # 時間長め
    
    result = predict_drowsiness(estimator, drowsy_sample, normalization_params)
    
    print("\n🔍 眠気状態サンプルの予測結果:")
    print(f"   予測: {result['class_name']}")
    print(f"   正常確率: {result['normal_probability']:.1%}")
    print(f"   眠気確率: {result['drowsy_probability']:.1%}")
    print(f"   信頼度: {result['confidence']:.1%}")
    
    print("\n" + "=" * 70)


def demo_with_dummy_data():
    """
    ダミーデータでのデモ
    """
    print("\n⚠️ 訓練済みモデルがないため、デモはスキップされます")
    print("   まず train_drowsiness_model.py でモデルを訓練してください")


def main():
    """
    メイン関数
    """
    print("=" * 70)
    print("📚 訓練済みモデルの使用方法サンプル")
    print("=" * 70)
    
    print("\nこのスクリプトは訓練済みモデルを読み込んで予測する方法を示します。")
    print("\n使用方法:")
    print("1. train_drowsiness_model.py でモデルを訓練")
    print("2. 生成された .pth ファイルのパスを指定")
    print("3. リアルタイムデータまたはテストデータで予測")
    
    demo_prediction()
    
    print("\n" + "=" * 70)
    print("💡 実際の使用例:")
    print("=" * 70)
    print("""
from src.lstm_drowsiness_model import DrowsinessEstimator
from src.blink_feature_extractor import BlinkFeatureExtractor

# モデル読み込み
estimator = DrowsinessEstimator()
estimator.load_model('models/trained_models/drowsiness_lstm_20240101_120000.pth')

# 特徴量抽出器
feature_extractor = BlinkFeatureExtractor(sequence_length=10)
feature_extractor.load_normalization_params('data/drowsiness_training_data/normalization_params.json')

# リアルタイムで瞬きデータを収集
# ... (blink_detectorなどで瞬き検出)

# 特徴量抽出
features = feature_extractor.extract_features(blink_data)

# シーケンス取得（10個溜まったら）
sequence = feature_extractor.get_sequence(normalize=True)

if sequence is not None:
    # 予測実行
    pred_class = estimator.predict(sequence[np.newaxis, ...])
    pred_proba = estimator.predict_proba(sequence[np.newaxis, ...])
    
    print(f"予測: {'眠気' if pred_class[0] == 1 else '正常'}")
    print(f"眠気確率: {pred_proba[0, 1]:.1%}")
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
