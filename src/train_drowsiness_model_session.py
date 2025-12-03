"""
眠気推定LSTMモデルの訓練スクリプト（セッション単位分割対応版）
Drowsiness Estimation LSTM Model Training Script with Session-based Splitting

データリークを防ぐため、セッション単位でデータを分割して訓練します。

使い方:
    python train_drowsiness_model_session.py --data-dir data --epochs 100
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Optional

# 現在のディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 内部モジュール
try:
    from drowsiness_data_manager import DrowsinessDataManager
    from src.lstm_drowsiness_model import DrowsinessEstimator
except ImportError:
    try:
        from src.drowsiness_data_manager import DrowsinessDataManager
        from src.lstm_drowsiness_model import DrowsinessEstimator
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        print("   必要なモジュールが見つかりません")
        sys.exit(1)

# 可視化
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # GUIなしで動作
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlibが見つかりません。グラフは生成されません。")


class ModelTrainerSessionBased:
    """
    眠気推定モデルのトレーナー（セッション単位分割対応版）
    
    データリークを防ぐため、同じセッション内のシーケンスは
    全て同じセット（訓練/検証/テスト）に配置されます。
    """
    
    def __init__(self, config: Dict):
        """
        初期化
        
        Args:
            config (Dict): 訓練設定
        """
        self.config = config
        self.data_manager = None
        self.estimator = None
        
        # 出力ディレクトリ
        self.output_dir = config.get('output_dir', 'trained_models')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        
        # モデル名（タイムスタンプ付き）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_name = f"drowsiness_lstm_{timestamp}"
        
        print("=" * 70)
        print("🎓 モデル訓練システム初期化（セッション単位分割対応版）")
        print("=" * 70)
        print(f"📁 出力ディレクトリ: {self.output_dir}")
        print(f"🏷️  モデル名: {self.model_name}")
    
    def load_data(self, data_dir: str) -> bool:
        """
        データを読み込み、セッション単位で分割
        
        Args:
            data_dir (str): データディレクトリ
            
        Returns:
            bool: 成功したかどうか
        """
        print("\n" + "=" * 70)
        print("📂 データ読み込み（セッション単位分割）")
        print("=" * 70)
        
        try:
            self.data_manager = DrowsinessDataManager(data_dir)
            
            # 生データから読み込み
            print("📊 生データから読み込み...")
            success = self.data_manager.load_all_data(verbose=True)
            
            if not success:
                print("❌ データ読み込み失敗")
                return False
            
            # セッション単位でデータ分割（データリーク防止）
            print("\n📊 セッション単位でデータ分割...")
            self.data_manager.split_data_by_session(
                train_ratio=self.config.get('train_ratio', 0.7),
                val_ratio=self.config.get('val_ratio', 0.15),
                test_ratio=self.config.get('test_ratio', 0.15),
                verbose=True
            )
            
            # 正規化
            print("\n🔧 データ正規化...")
            self.data_manager.normalize_data(
                method=self.config.get('normalization', 'zscore'),
                verbose=True
            )
            
            # データセット保存
            dataset_file = os.path.join(data_dir, 'drowsiness_dataset_session_split.npz')
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
            'input_size': self.config.get('input_size', 12),
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
    
    def train_model(self) -> bool:
        """
        モデルを訓練
        
        Returns:
            bool: 成功したかどうか
        """
        if self.data_manager is None:
            print("❌ データが読み込まれていません")
            return False
        
        print("\n" + "=" * 70)
        print("🚀 モデル訓練開始")
        print("=" * 70)
        
        # データ取得
        train_sequences, train_labels = self.data_manager.get_train_data()
        val_sequences, val_labels = self.data_manager.get_val_data()
        
        if len(train_sequences) == 0:
            print("❌ 訓練データがありません")
            return False
        
        # 訓練
        import time
        start_time = time.time()
        
        history = self.estimator.train_model(
            train_sequences=train_sequences,
            train_labels=train_labels,
            val_sequences=val_sequences,
            val_labels=val_labels,
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 32),
            learning_rate=self.config.get('learning_rate', 0.001),
            patience=self.config.get('patience', 10),
            verbose=True
        )
        
        self.training_time = time.time() - start_time
        
        print(f"\n⏱️  訓練時間: {self.training_time:.1f}秒")
        print(f"🎯 ベスト検証精度: {max(history['val_acc']):.2f}%")
        
        return True
    
    def plot_training_history(self):
        """訓練履歴を可視化"""
        if not HAS_MATPLOTLIB:
            return
        
        print("\n" + "=" * 70)
        print("📊 訓練履歴の可視化")
        print("=" * 70)
        
        history = self.estimator.history
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 損失
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss (Session-based Split)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 精度
        axes[1].plot(history['train_acc'], label='Train Acc')
        axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy (Session-based Split)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        plot_path = os.path.join(
            self.output_dir, 'logs', f"{self.model_name}_history.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 訓練履歴グラフ保存: {plot_path}")
    
    def evaluate_model(self) -> Optional[Dict]:
        """
        モデルを評価
        
        Returns:
            Dict: 評価結果
        """
        print("\n" + "=" * 70)
        print("📊 モデル評価")
        print("=" * 70)
        
        test_sequences, test_labels = self.data_manager.get_test_data()
        
        if len(test_sequences) == 0:
            print("⚠️ テストデータがありません")
            return None
        
        # 予測
        predictions = self.estimator.predict(test_sequences)
        
        # 精度計算
        accuracy = np.mean(predictions == test_labels) * 100
        
        # 混同行列
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(test_labels, predictions)
        
        # 分類レポート
        target_names = ['Normal', 'Drowsy']
        report = classification_report(
            test_labels, predictions,
            target_names=target_names,
            output_dict=True
        )
        
        print(f"\n📊 評価結果:")
        print(f"   正解率: {accuracy:.2f}%")
        print(f"\n混同行列:")
        print(f"              予測: Normal  Drowsy")
        print(f"   実際: Normal    {cm[0][0]:5d}   {cm[0][1]:5d}")
        print(f"         Drowsy    {cm[1][0]:5d}   {cm[1][1]:5d}")
        
        print(f"\n📈 詳細な分類レポート:")
        for class_name in target_names:
            print(f"\n{class_name}:")
            print(f"  適合率 (Precision): {report[class_name]['precision']:.3f}")
            print(f"  再現率 (Recall):    {report[class_name]['recall']:.3f}")
            print(f"  F1スコア:          {report[class_name]['f1-score']:.3f}")
            print(f"  サポート:          {report[class_name]['support']}")
        
        print(f"\nマクロ平均:")
        print(f"  適合率: {report['macro avg']['precision']:.3f}")
        print(f"  再現率: {report['macro avg']['recall']:.3f}")
        print(f"  F1スコア: {report['macro avg']['f1-score']:.3f}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, results: Dict):
        """混同行列を可視化"""
        if not HAS_MATPLOTLIB:
            return
        
        print("\n📊 混同行列の可視化")
        
        cm = np.array(results['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        classes = ['Normal', 'Drowsy']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title='Confusion Matrix (Session-based Split)',
               ylabel='True label',
               xlabel='Predicted label')
        
        # 数値を表示
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        cm_path = os.path.join(
            self.output_dir, 'logs', f"{self.model_name}_confusion_matrix.png"
        )
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 混同行列グラフ保存: {cm_path}")
    
    def save_model(self, results: Optional[Dict] = None):
        """モデルを保存"""
        print("\n" + "=" * 70)
        print("💾 モデル保存")
        print("=" * 70)
        
        # モデル保存
        model_path = os.path.join(self.output_dir, f"{self.model_name}.pth")
        self.estimator.save_model(model_path)
        print(f"✅ モデル保存: {model_path}")
        
        # メタデータ保存
        metadata = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'training_time': getattr(self, 'training_time', 0),
            'best_val_acc': max(self.estimator.history['val_acc']),
            'config': self.config,
            'split_method': 'session-based',
            'train_sessions': [s['name'] for s in self.data_manager.train_sessions],
            'val_sessions': [s['name'] for s in self.data_manager.val_sessions],
            'test_sessions': [s['name'] for s in self.data_manager.test_sessions],
            'data_statistics': {
                'train_count': len(self.data_manager.train_sequences),
                'val_count': len(self.data_manager.val_sequences),
                'test_count': len(self.data_manager.test_sequences)
            }
        }
        
        if results:
            metadata['test_accuracy'] = results['accuracy']
            metadata['confusion_matrix'] = results['confusion_matrix']
            metadata['classification_report'] = results['classification_report']
        
        metadata_path = os.path.join(self.output_dir, f"{self.model_name}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ メタデータ保存: {metadata_path}")
    
    def run_full_training_pipeline(self, data_dir: str) -> bool:
        """
        完全な訓練パイプラインを実行
        
        Args:
            data_dir (str): データディレクトリ
            
        Returns:
            bool: 成功したかどうか
        """
        print("\n" + "=" * 70)
        print("🎓 完全訓練パイプライン開始（セッション単位分割）")
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
        if results:
            print(f"🎯 最終テスト精度: {results['accuracy']:.2f}%")
        print(f"📁 モデル: {self.output_dir}/{self.model_name}.pth")
        print("=" * 70)
        
        return True


def create_default_config() -> Dict:
    """デフォルト設定を作成"""
    return {
        # データ設定
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'normalization': 'zscore',
        
        # モデル設定
        'input_size': 12,
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
        'device': None
    }


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='眠気推定LSTMモデルの訓練（セッション単位分割対応版）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # データ設定
    parser.add_argument('--data-dir', type=str, default='data',
                       help='データディレクトリ')
    parser.add_argument('--output-dir', type=str, default='trained_models',
                       help='出力ディレクトリ')
    
    # モデル設定
    parser.add_argument('--input-size', type=int, default=12,
                       help='入力次元')
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
                       help='使用デバイス (cuda/cpu)')
    parser.add_argument('--show-plots', action='store_true',
                       help='グラフを表示')
    
    return parser.parse_args()


def main():
    """メイン関数"""
    args = parse_args()
    
    # 設定作成
    config = create_default_config()
    
    # コマンドライン引数で上書き
    config['output_dir'] = args.output_dir
    config['input_size'] = args.input_size
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
    
    print("=" * 70)
    print("🎓 眠気推定モデル訓練システム（セッション単位分割対応版）")
    print("=" * 70)
    print("\n📋 訓練設定:")
    print(json.dumps(config, indent=2))
    
    # トレーナー作成
    trainer = ModelTrainerSessionBased(config)
    
    # 訓練実行
    success = trainer.run_full_training_pipeline(args.data_dir)
    
    if success:
        print("\n🎉 訓練が正常に完了しました！")
    else:
        print("\n❌ 訓練に失敗しました")
        sys.exit(1)


if __name__ == "__main__":
    main()
