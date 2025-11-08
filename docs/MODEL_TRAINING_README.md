# 段階5: モデル訓練システム

## 作成されたファイル

1. **train_drowsiness_model.py** - モデル訓練スクリプト
2. **training_config.json** - 設定ファイルサンプル

---

## 🎓 train_drowsiness_model.py

### 機能

- データの自動読み込み・前処理
- LSTMモデルの訓練
- Early Stopping機能
- 訓練履歴の可視化
- モデル評価（混同行列、分類レポート）
- 訓練済みモデルの保存

---

## 📖 使用方法

### 基本的な使い方

```bash
# デフォルト設定で訓練
python train_drowsiness_model.py --data-dir drowsiness_training_data

# エポック数を指定
python train_drowsiness_model.py --data-dir drowsiness_training_data --epochs 150

# バッチサイズと学習率を指定
python train_drowsiness_model.py --data-dir drowsiness_training_data \
    --batch-size 64 --lr 0.0005

# グラフを表示
python train_drowsiness_model.py --data-dir drowsiness_training_data --show-plots
```

### 設定ファイルを使用

```bash
# 設定ファイルから読み込み
python train_drowsiness_model.py --config training_config.json \
    --data-dir drowsiness_training_data
```

### コマンドライン引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--data-dir` | `drowsiness_training_data` | データディレクトリ |
| `--output-dir` | `trained_models` | モデル出力ディレクトリ |
| `--hidden-size1` | `64` | LSTM第1層のユニット数 |
| `--hidden-size2` | `32` | LSTM第2層のユニット数 |
| `--fc-size` | `32` | 全結合層のユニット数 |
| `--dropout` | `0.3` | ドロップアウト率 |
| `--epochs` | `100` | エポック数 |
| `--batch-size` | `32` | バッチサイズ |
| `--lr` | `0.001` | 学習率 |
| `--patience` | `10` | Early Stoppingの忍耐値 |
| `--device` | `None` | 使用デバイス（cuda/cpu） |
| `--show-plots` | `False` | グラフ表示 |
| `--config` | `None` | 設定ファイル（JSON） |

---

## 📊 出力ファイル

### ディレクトリ構造

```
trained_models/
├── drowsiness_lstm_20240101_120000.pth           # 訓練済みモデル
├── drowsiness_lstm_20240101_120000_metadata.json # メタデータ
└── logs/
    ├── drowsiness_lstm_20240101_120000_history.png        # 訓練履歴
    └── drowsiness_lstm_20240101_120000_confusion_matrix.png # 混同行列
```

### モデルファイル (.pth)

PyTorch形式の訓練済みモデル。以下の情報を含む：
- モデルの重み（state_dict）
- モデルパラメータ（アーキテクチャ設定）
- 訓練履歴（損失、精度）

### メタデータ (.json)

訓練に関する詳細情報：

```json
{
  "model_name": "drowsiness_lstm_20240101_120000",
  "timestamp": "20240101_120000",
  "training_time": 120.5,
  "best_val_acc": 92.5,
  "test_accuracy": 91.2,
  "config": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    ...
  },
  "data_statistics": {
    "total_sequences": 500,
    "normal_count": 250,
    "drowsy_count": 250,
    ...
  },
  "confusion_matrix": [[45, 5], [3, 47]],
  "classification_report": {
    "正常": {
      "precision": 0.938,
      "recall": 0.900,
      "f1-score": 0.918
    },
    "眠気": {
      "precision": 0.904,
      "recall": 0.940,
      "f1-score": 0.922
    }
  }
}
```

### 訓練履歴グラフ

訓練損失・検証損失、訓練精度・検証精度の推移をプロット

### 混同行列グラフ

テストデータでの予測結果を混同行列で表示

---

## 🔄 訓練パイプライン

```
1. データ読み込み
   ↓
   drowsiness_training_data/ から全データを読み込み
   既存のdrowsiness_dataset.npzがあればそれを使用
   
2. データ前処理
   ↓
   訓練/検証/テスト分割（70%/15%/15%）
   Z-score正規化
   
3. モデル作成
   ↓
   LSTMアーキテクチャの初期化
   
4. モデル訓練
   ↓
   バッチ学習
   Early Stopping
   検証データで評価
   
5. 可視化
   ↓
   訓練履歴グラフ作成
   混同行列グラフ作成
   
6. 評価
   ↓
   テストデータで最終評価
   分類レポート出力
   
7. 保存
   ↓
   モデル、メタデータ、グラフを保存
```

---

## 🎯 ハイパーパラメータチューニング

### エポック数

```bash
# 短い訓練（テスト用）
python train_drowsiness_model.py --epochs 30

# 長い訓練（高精度狙い）
python train_drowsiness_model.py --epochs 200
```

### バッチサイズ

```bash
# 小さいバッチ（細かい更新）
python train_drowsiness_model.py --batch-size 16

# 大きいバッチ（高速訓練）
python train_drowsiness_model.py --batch-size 64
```

### 学習率

```bash
# 低い学習率（安定した学習）
python train_drowsiness_model.py --lr 0.0005

# 高い学習率（速い学習）
python train_drowsiness_model.py --lr 0.002
```

### モデルサイズ

```bash
# 小さいモデル（高速・過学習しにくい）
python train_drowsiness_model.py --hidden-size1 32 --hidden-size2 16 --fc-size 16

# 大きいモデル（高精度・過学習しやすい）
python train_drowsiness_model.py --hidden-size1 128 --hidden-size2 64 --fc-size 64
```

### ドロップアウト

```bash
# 低いドロップアウト（表現力重視）
python train_drowsiness_model.py --dropout 0.2

# 高いドロップアウト（汎化性能重視）
python train_drowsiness_model.py --dropout 0.5
```

---

## 💡 訓練のコツ

### 1. データ量

- **推奨**: 各クラス最低100シーケンス以上
- データが少ない場合は、ドロップアウト率を上げる

### 2. クラスバランス

- 正常と眠気のデータ数を同程度にする
- 不均衡な場合は、データ収集を追加

### 3. 過学習の兆候

- 訓練精度は高いが検証精度が低い
- 対策:
  - ドロップアウト率を上げる（0.3 → 0.5）
  - データを増やす
  - モデルサイズを小さくする

### 4. 未学習の兆候

- 訓練精度も検証精度も低い
- 対策:
  - エポック数を増やす
  - 学習率を調整
  - モデルサイズを大きくする

### 5. Early Stoppingの調整

```bash
# 早めに停止（過学習防止）
python train_drowsiness_model.py --patience 5

# 長く訓練（精度向上）
python train_drowsiness_model.py --patience 20
```

---

## 📈 訓練結果の解釈

### 訓練履歴グラフ

**理想的なパターン:**
- 訓練損失と検証損失が同時に下がる
- 訓練精度と検証精度が同時に上がる
- 収束している

**過学習のパターン:**
- 訓練損失は下がるが検証損失は上がる
- 訓練精度と検証精度の差が大きい

**未学習のパターン:**
- 両方の損失が高い
- 両方の精度が低い

### 混同行列

```
              予測: 正常  眠気
実際: 正常       45      5
      眠気        3     47
```

- **正常の再現率**: 45/(45+5) = 90% → 正常状態の90%を正しく検出
- **眠気の再現率**: 47/(3+47) = 94% → 眠気状態の94%を正しく検出
- **正常の適合率**: 45/(45+3) = 94% → 正常と予測したうち94%が正解
- **眠気の適合率**: 47/(5+47) = 90% → 眠気と予測したうち90%が正解

---

## 🔧 トラブルシューティング

### データが見つからない

```
❌ シーケンスデータが見つかりません
```

→ `drowsiness_data_collector.py` でデータを収集してください

### メモリ不足

```
RuntimeError: CUDA out of memory
```

→ バッチサイズを小さくする: `--batch-size 16`

### 訓練が遅い

- GPU使用を確認: `--device cuda`
- バッチサイズを大きくする: `--batch-size 64`
- エポック数を減らす: `--epochs 50`

### 精度が低い

- データ量を増やす
- エポック数を増やす
- ハイパーパラメータを調整
- データの品質を確認

---

## 🐍 Pythonスクリプトとして使用

```python
from train_drowsiness_model import ModelTrainer, create_default_config

# 設定作成
config = create_default_config()
config['epochs'] = 150
config['batch_size'] = 64

# トレーナー作成
trainer = ModelTrainer(config)

# 訓練実行
trainer.run_full_training_pipeline('drowsiness_training_data')
```

---

## 📝 次のステップ

訓練が完了したら：

1. **モデル評価**
   - テスト精度を確認
   - 混同行列を分析
   - 必要に応じて再訓練

2. **段階6: リアルタイム推定システムの構築**
   - 訓練済みモデルを使用
   - カメラからリアルタイムで眠気を推定

---

## 💡 よくある質問

**Q: 訓練にどのくらい時間がかかりますか？**  
A: データ量とハードウェアに依存します。目安：
- CPU: 500シーケンス、100エポックで約5-10分
- GPU: 同条件で約1-2分

**Q: 何エポック訓練すればいいですか？**  
A: Early Stoppingが効くため、多めに設定（100-200）して自動停止に任せるのが良いです。

**Q: どのくらいの精度が出れば良いですか？**  
A: テスト精度85%以上が目標です。90%以上なら優秀です。

**Q: GPU は必須ですか？**  
A: 必須ではありませんが、訓練が5-10倍速くなります。

**Q: 訓練に失敗しました**  
A: エラーメッセージを確認し、以下を確認：
- データが正しく収集されているか
- 必要なライブラリがインストールされているか
- メモリは十分か
