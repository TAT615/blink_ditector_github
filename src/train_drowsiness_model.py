"""
眠気推定モデル訓練スクリプト
Drowsiness Model Training Script

収集したデータを使ってLSTMモデルを訓練します。
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

# 自作モジュールのインポート
try:
    from src.drowsiness_data_manager import DrowsinessDataManager
    from src.lstm_drowsiness_model import DrowsinessEstimator
except ImportError as e:
    print(f"❌ モジュールのインポートエラー: {e}")
    print("   必要なファイル: src/drowsiness_data_manager.py, src/lstm_drowsiness_model.py")
    print("   プロジェクトルートから実行してください: python -m src.train_drowsiness_model")
    sys.exit(1)


class ModelTrainer:
    """
    モデル訓練を管理するクラス
    """
    
    def __init__(self, config: Dict):
        """
        初期化
        
        Args:
            config (Dict): 設定パラメータ
        """
        self.config = config
        
        # ディレクトリ設定
        self.output_dir = config.get('output_dir', 'trained_models')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # タイムスタンプ
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f"drowsiness_lstm_{self.timestamp}"
        
        # データマネージャー
        self.data_manager = None
        
        # モデル推定器
        self.estimator = None
        
        # 訓練履歴
        self.history = None
        self.training_time = 0
        self.best_val_acc = 0
        
        print("=" * 70)
        print("🎓 モデル訓練システム初期化")
        print("=" * 70)
        print(f"📁 出力ディレクトリ: {self.output_dir}")
        print(f"🏷️  モデル名: {self.model_name}")
    
    def load_data(self, data_dir: str) -> bool:
        """
        データを読み込み
        
        Args:
            data_dir (str): データディレクトリ
            
        Returns:
            bool: 成功したかどうか
        """
        print("\n" + "=" * 70)
        print("📂 データ読み込み")
        print("=" * 70)
        
        try:
            # データマネージャー作成
            self.data_manager = DrowsinessDataManager(data_dir=data_dir)
            
            # データセットファイルがあればそれを使用
            dataset_file = os.path.join(data_dir, 'drowsiness_dataset.npz')
            if os.path.exists(dataset_file):
                print(f"📦 既存のデータセットを読み込み: {dataset_file}")
                self.data_manager.load_dataset(dataset_file)
                return True
            
            # なければ生データから読み込み
            print("📊 生データから読み込み...")
            success = self.data_manager.load_all_data(verbose=True)
            
            if not success:
                print("❌ データ読み込み失敗")
                return False
            
            # データ分割
            print("\n📊 データ分割...")
            self.data_manager.split_data(
                train_ratio=self.config.get('train_ratio', 0.7),
                val_ratio=self.config.get('val_ratio', 0.15),
                test_ratio=self.config.get('test_ratio', 0.15),
                stratify=True,
                verbose=True
            )
            
            # 正規化
            print("\n🔧 データ正規化...")
            self.data_manager.normalize_data(
                method=self.config.get('normalization', 'zscore'),
                verbose=True
            )
            
            # データセット保存
            self.data_manager.export_dataset(dataset_file)
            
            # 正規化パラメータ保存
            norm_params_file = os.path.join(data_dir, 'normalization_params.json')
            self.data_manager.save_normalization_params(norm_params_file)
            
            return True
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_model(self):
        """
        モデルを作成
        """
        print("\n" + "=" * 70)
        print("🧠 モデル作成")
        print("=" * 70)
        
        # モデルパラメータ
        model_params = {
            'input_size': self.config.get('input_size', 6),
            'hidden_size1': self.config.get('hidden_size1', 64),
            'hidden_size2': self.config.get('hidden_size2', 32),
            'fc_size': self.config.get('fc_size', 32),
            'num_classes': self.config.get('num_classes', 2),
            'dropout_rate': self.config.get('dropout_rate', 0.3)
        }
        
        # 推定器作成
        self.estimator = DrowsinessEstimator(
            model_params=model_params,
            device=self.config.get('device', None)
        )
        
        self.estimator.get_model_summary()
    
    def train_model(self):
        """
        モデルを訓練
        """
        if self.data_manager is None:
            print("❌ データが読み込まれていません")
            return False
        
        if self.estimator is None:
            print("❌ モデルが作成されていません")
            return False
        
        print("\n" + "=" * 70)
        print("🚀 モデル訓練開始")
        print("=" * 70)
        
        # 訓練データ取得
        train_sequences, train_labels = self.data_manager.get_train_data()
        val_sequences, val_labels = self.data_manager.get_val_data()
        
        # 訓練パラメータ
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        learning_rate = self.config.get('learning_rate', 0.001)
        patience = self.config.get('patience', 10)
        
        # 訓練実行
        start_time = time.time()
        
        self.history = self.estimator.train_model(
            train_sequences, train_labels,
            val_sequences, val_labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            verbose=True
        )
        
        self.training_time = time.time() - start_time
        
        # ベスト検証精度を記録
        if len(self.history['val_acc']) > 0:
            self.best_val_acc = max(self.history['val_acc'])
        
        print(f"\n⏱️  訓練時間: {self.training_time:.1f}秒")
        print(f"🎯 ベスト検証精度: {self.best_val_acc:.2f}%")
        
        return True
    
    def evaluate_model(self):
        """
        モデルを評価
        """
        if self.estimator is None:
            print("❌ モデルが訓練されていません")
            return None
        
        print("\n" + "=" * 70)
        print("📊 モデル評価")
        print("=" * 70)
        
        # テストデータ取得
        test_sequences, test_labels = self.data_manager.get_test_data()
        
        # 評価実行
        results = self.estimator.evaluate(test_sequences, test_labels)
        
        # 詳細な分類レポート
        print("\n📈 詳細な分類レポート:")
        report = results['classification_report']
        
        for class_name in ['正常', '眠気']:
            if class_name in report:
                metrics = report[class_name]
                print(f"\n{class_name}:")
                print(f"  適合率 (Precision): {metrics['precision']:.3f}")
                print(f"  再現率 (Recall):    {metrics['recall']:.3f}")
                print(f"  F1スコア:          {metrics['f1-score']:.3f}")
                print(f"  サポート:          {metrics['support']}")
        
        # マクロ平均
        if 'macro avg' in report:
            macro = report['macro avg']
            print(f"\nマクロ平均:")
            print(f"  適合率: {macro['precision']:.3f}")
            print(f"  再現率: {macro['recall']:.3f}")
            print(f"  F1スコア: {macro['f1-score']:.3f}")
        
        return results
    
    def save_model(self, results: Optional[Dict] = None):
        """
        モデルとメタデータを保存
        
        Args:
            results (Dict): 評価結果
        """
        print("\n" + "=" * 70)
        print("💾 モデル保存")
        print("=" * 70)
        
        # モデルファイルパス
        model_path = os.path.join(self.output_dir, f"{self.model_name}.pth")
        
        # モデル保存
        self.estimator.save_model(model_path, include_history=True)
        
        # メタデータ作成
        metadata = {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'training_time': self.training_time,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'data_statistics': self.data_manager.get_statistics()
        }
        
        if results is not None:
            metadata['test_accuracy'] = results['accuracy']
            metadata['confusion_matrix'] = results['confusion_matrix']
            metadata['classification_report'] = results['classification_report']
        
        # メタデータ保存
        metadata_path = os.path.join(self.output_dir, f"{self.model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ モデル保存: {model_path}")
        print(f"✅ メタデータ保存: {metadata_path}")
    
    def plot_training_history(self):
        """
        訓練履歴をプロット
        """
        if self.history is None:
            print("⚠️ 訓練履歴がありません")
            return
        
        print("\n" + "=" * 70)
        print("📊 訓練履歴の可視化")
        print("=" * 70)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 損失のプロット
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        
        axes[0].plot(epochs_range, self.history['train_loss'], 'b-', label='訓練損失', linewidth=2)
        if len(self.history['val_loss']) > 0:
            axes[0].plot(epochs_range, self.history['val_loss'], 'r-', label='検証損失', linewidth=2)
        axes[0].set_xlabel('エポック', fontsize=12)
        axes[0].set_ylabel('損失', fontsize=12)
        axes[0].set_title('訓練損失と検証損失', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # 精度のプロット
        axes[1].plot(epochs_range, self.history['train_acc'], 'b-', label='訓練精度', linewidth=2)
        if len(self.history['val_acc']) > 0:
            axes[1].plot(epochs_range, self.history['val_acc'], 'r-', label='検証精度', linewidth=2)
        axes[1].set_xlabel('エポック', fontsize=12)
        axes[1].set_ylabel('精度 (%)', fontsize=12)
        axes[1].set_title('訓練精度と検証精度', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        plot_path = os.path.join(self.log_dir, f"{self.model_name}_history.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ 訓練履歴グラフ保存: {plot_path}")
        
        # 表示（オプション）
        if self.config.get('show_plots', False):
            plt.show()
        else:
            plt.close()
    
    def plot_confusion_matrix(self, results: Dict):
        """
        混同行列をプロット
        
        Args:
            results (Dict): 評価結果
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        
        print("\n📊 混同行列の可視化")
        
        cm = np.array(results['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['正常', '眠気']
        )
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title('混同行列', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        cm_path = os.path.join(self.log_dir, f"{self.model_name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"✅ 混同行列グラフ保存: {cm_path}")
        
        if self.config.get('show_plots', False):
            plt.show()
        else:
            plt.close()
    
    def run_full_training_pipeline(self, data_dir: str):
        """
        完全な訓練パイプラインを実行
        
        Args:
            data_dir (str): データディレクトリ
            
        Returns:
            bool: 成功したかどうか
        """
        print("\n" + "=" * 70)
        print("🎓 完全訓練パイプライン開始")
        print("=" * 70)
        
        # 1. データ読み込み
        if not self.load_data(data_dir):
            return False
        
        # 2. モデル作成
        self.create_model()
        
        # 3. モデル訓練
        if not self.train_model():
            return False
        
        # 4. 訓練履歴可視化
        self.plot_training_history()
        
        # 5. モデル評価
        results = self.evaluate_model()
        
        # 6. 混同行列可視化
        if results is not None:
            self.plot_confusion_matrix(results)
        
        # 7. モデル保存
        self.save_model(results)
        
        print("\n" + "=" * 70)
        print("✅ 訓練パイプライン完了")
        print("=" * 70)
        print(f"🎯 最終テスト精度: {results['accuracy']:.2f}%")
        print(f"📁 モデル: {self.output_dir}/{self.model_name}.pth")
        print("=" * 70)
        
        return True


def create_default_config() -> Dict:
    """
    デフォルト設定を作成
    
    Returns:
        Dict: デフォルト設定
    """
    return {
        # データ設定
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'normalization': 'zscore',
        
        # モデル設定
        'input_size': 6,
        'hidden_size1': 64,
        'hidden_size2': 32,
        'fc_size': 32,
        'num_classes': 2,
        'dropout_rate': 0.3,
        
        # 訓練設定
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'patience': 10,
        
        # 出力設定
        'output_dir': 'trained_models',
        'show_plots': False,
        'device': None  # None=自動選択, 'cuda', 'cpu'
    }


def parse_args():
    """
    コマンドライン引数をパース
    
    Returns:
        argparse.Namespace: パースされた引数
    """
    parser = argparse.ArgumentParser(
        description='眠気推定LSTMモデルの訓練',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # データ設定
    parser.add_argument('--data-dir', type=str, default='drowsiness_training_data',
                       help='データディレクトリ')
    parser.add_argument('--output-dir', type=str, default='trained_models',
                       help='モデル出力ディレクトリ')
    
    # モデル設定
    parser.add_argument('--hidden-size1', type=int, default=64,
                       help='LSTM第1層のユニット数')
    parser.add_argument('--hidden-size2', type=int, default=32,
                       help='LSTM第2層のユニット数')
    parser.add_argument('--fc-size', type=int, default=32,
                       help='全結合層のユニット数')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='ドロップアウト率')
    
    # 訓練設定
    parser.add_argument('--epochs', type=int, default=100,
                       help='エポック数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学習率')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early Stoppingの忍耐値')
    
    # その他
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu', None],
                       help='使用デバイス')
    parser.add_argument('--show-plots', action='store_true',
                       help='グラフを表示する')
    parser.add_argument('--config', type=str, default=None,
                       help='設定ファイル（JSON）')
    
    return parser.parse_args()


def load_config_from_file(filepath: str) -> Dict:
    """
    設定ファイルを読み込み
    
    Args:
        filepath (str): 設定ファイルパス
        
    Returns:
        Dict: 設定
    """
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        print(f"✅ 設定ファイル読み込み: {filepath}")
        return config
    except Exception as e:
        print(f"❌ 設定ファイル読み込みエラー: {e}")
        return {}


def main():
    """
    メイン関数
    """
    print("=" * 70)
    print("🎓 眠気推定モデル訓練システム")
    print("=" * 70)
    
    # 引数パース
    args = parse_args()
    
    # 設定作成
    if args.config is not None:
        config = load_config_from_file(args.config)
    else:
        config = create_default_config()
    
    # コマンドライン引数で上書き
    config['output_dir'] = args.output_dir
    config['hidden_size1'] = args.hidden_size1
    config['hidden_size2'] = args.hidden_size2
    config['fc_size'] = args.fc_size
    config['dropout_rate'] = args.dropout
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['patience'] = args.patience
    config['device'] = args.device
    config['show_plots'] = args.show_plots
    
    # 設定表示
    print("\n📋 訓練設定:")
    print(json.dumps(config, indent=2))
    
    # トレーナー作成
    trainer = ModelTrainer(config)
    
    # 訓練実行
    success = trainer.run_full_training_pipeline(args.data_dir)
    
    if success:
        print("\n🎉 訓練が正常に完了しました！")
        sys.exit(0)
    else:
        print("\n❌ 訓練に失敗しました")
        sys.exit(1)


if __name__ == "__main__":
    main()
