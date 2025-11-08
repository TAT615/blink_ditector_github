# 眠気推定システム開発 - 進捗状況

## 📊 開発段階の進捗

### ✅ 段階1: 基盤モジュール（既存）
- `blink_detector.py` - 4段階EAR検出システム
- 個人適応型キャリブレーション機能

### ✅ 段階2: 特徴量抽出モジュール
**作成ファイル:**
- `blink_feature_extractor.py` - 瞬き係数と6次元特徴量抽出

**機能:**
- 瞬き係数 (To/Tc) の計算
- 6次元特徴ベクトル生成
- 正規化機能
- シーケンス生成

### ✅ 段階3: LSTMモデル
**作成ファイル:**
- `lstm_drowsiness_model.py` - LSTM分類モデル

**機能:**
- 2層LSTM + 全結合層アーキテクチャ
- 訓練・推論機能
- Early Stopping
- モデル保存・読み込み

**パラメータ数:** 31,714

### ✅ 段階4: データ収集・管理
**作成ファイル:**
- `drowsiness_data_collector.py` - インタラクティブデータ収集
- `drowsiness_data_manager.py` - データ管理・前処理
- `DATA_COLLECTION_README.md` - 使用方法ドキュメント

**機能:**
- リアルタイム瞬き検出と特徴量抽出
- 正常/眠気状態のラベリング
- データの保存（CSV/JSON/NumPy）
- データ分割・正規化

### ✅ 段階5: モデル訓練
**作成ファイル:**
- `train_drowsiness_model.py` - メイン訓練スクリプト
- `training_config.json` - 設定ファイルサンプル
- `MODEL_TRAINING_README.md` - 詳細な使用方法
- `test_training_script.py` - 動作確認テスト
- `load_and_use_model.py` - モデル使用サンプル

**機能:**
- 自動データ読み込み・前処理
- モデル訓練とEarly Stopping
- 訓練履歴の可視化
- モデル評価（混同行列、分類レポート）
- コマンドライン引数対応

### ⬜ 段階6: リアルタイム推定システム
**予定:**
- `realtime_drowsiness_estimator.py` - 統合システム

---

## 🗂️ ファイル一覧

### コアモジュール
```
blink_feature_extractor.py       # 特徴量抽出
lstm_drowsiness_model.py         # LSTMモデル
drowsiness_data_collector.py     # データ収集
drowsiness_data_manager.py       # データ管理
train_drowsiness_model.py        # モデル訓練
```

### サポートファイル
```
training_config.json             # 訓練設定サンプル
test_training_script.py          # 動作テスト
load_and_use_model.py           # モデル使用サンプル
```

### ドキュメント
```
DATA_COLLECTION_README.md        # データ収集ガイド
MODEL_TRAINING_README.md         # モデル訓練ガイド
```

---

## 🚀 使用フロー

### 1. データ収集
```bash
python drowsiness_data_collector.py
```
- [N]キーで正常状態のデータ収集
- [D]キーで眠気状態のデータ収集
- 各状態で10セッション以上推奨

### 2. データ確認
```python
from drowsiness_data_manager import DrowsinessDataManager

manager = DrowsinessDataManager()
manager.load_all_data()
manager.print_statistics()
```

### 3. モデル訓練
```bash
python train_drowsiness_model.py --data-dir drowsiness_training_data --epochs 100
```

### 4. モデル評価
訓練完了後、以下が生成されます：
- `trained_models/drowsiness_lstm_YYYYMMDD_HHMMSS.pth`
- メタデータ（JSON）
- 訓練履歴グラフ
- 混同行列

### 5. モデル使用
```python
from lstm_drowsiness_model import DrowsinessEstimator

estimator = DrowsinessEstimator()
estimator.load_model('trained_models/drowsiness_lstm_20240101_120000.pth')

# 予測
predictions = estimator.predict(test_sequences)
probabilities = estimator.predict_proba(test_sequences)
```

---

## 📦 データ構造

### 6次元特徴ベクトル
1. **瞬き係数 (To/Tc)** - 開眼速度と閉眼速度の比率
2. **閉眼時間 Tc [秒]** - まぶたを閉じる時間
3. **開眼時間 To [秒]** - まぶたを開く時間
4. **瞬き間隔 [秒]** - 前回の瞬きからの時間
5. **EAR最小値** - 瞬き時のEAR最小値
6. **総瞬き時間 [秒]** - Tc + To

### シーケンスデータ
- **形状**: `(n_samples, 10, 6)`
- **説明**: 過去10回の瞬きデータ
- **ラベル**: 0=正常, 1=眠気

---

## 🎯 システムアーキテクチャ

```
カメラ入力
  ↓
[blink_detector.py]
4段階EAR検出
  ↓
瞬きデータ (t1, t2, t3, ear_min)
  ↓
[blink_feature_extractor.py]
特徴量抽出 → 6次元ベクトル
  ↓
シーケンス生成 (10個の瞬き)
  ↓
[lstm_drowsiness_model.py]
LSTM推論
  ↓
予測結果 (正常/眠気)
```

---

## 🧠 LSTMモデル詳細

### アーキテクチャ
```
入力: (batch, 10, 6)
  ↓
LSTM層1: 64ユニット + Dropout(0.3)
  ↓
LSTM層2: 32ユニット + Dropout(0.3)
  ↓
全結合層: 32ユニット + ReLU + Dropout(0.3)
  ↓
出力層: 2クラス (Softmax)
```

### パラメータ
- 総パラメータ数: 31,714
- 最適化: Adam (lr=0.001)
- 損失関数: Cross Entropy
- Early Stopping: patience=10

---

## 📈 推奨パラメータ

### データ収集
- 最小セッション数: 各クラス10セッション
- 推奨セッション数: 各クラス20セッション以上
- セッション長: 30-60秒
- 最小瞬き数/セッション: 10回以上

### 訓練設定
- エポック数: 100-200
- バッチサイズ: 16-64
- 学習率: 0.0005-0.002
- ドロップアウト: 0.2-0.5

### 期待精度
- 目標テスト精度: 85%以上
- 優秀な精度: 90%以上

---

## 🔧 必要な依存ライブラリ

```
numpy
opencv-python (cv2)
dlib
torch (PyTorch)
scikit-learn
matplotlib
```

インストール:
```bash
pip install numpy opencv-python dlib torch scikit-learn matplotlib --break-system-packages
```

---

## 💡 次のステップ

### 段階6: リアルタイム推定システム
以下の機能を統合：
1. カメラからのリアルタイム入力
2. 瞬き検出と特徴量抽出
3. LSTM推論
4. 結果の可視化
5. アラート機能

**予定ファイル:**
- `realtime_drowsiness_estimator.py`

---

## 🐛 トラブルシューティング

### データ収集がうまくいかない
- カメラの接続を確認
- 照明条件を改善
- キャリブレーションを再実行

### 訓練が失敗する
- データが十分に収集されているか確認
- メモリを確認（バッチサイズを減らす）
- 依存ライブラリのインストール確認

### 精度が低い
- データ量を増やす（各クラス100シーケンス以上）
- クラスバランスを確認
- ハイパーパラメータを調整
- データの品質を確認（キャリブレーション）

---

## 📚 ドキュメント

- **データ収集**: `DATA_COLLECTION_README.md`
- **モデル訓練**: `MODEL_TRAINING_README.md`
- **論文参照**: プロジェクト内の論文PDFファイル

---

## ✅ 達成済み機能

- [x] 4段階EAR検出
- [x] 瞬き係数計算
- [x] 6次元特徴量抽出
- [x] LSTMモデル実装
- [x] データ収集システム
- [x] データ管理システム
- [x] モデル訓練システム
- [x] 訓練履歴可視化
- [x] モデル評価機能
- [ ] リアルタイム推定システム

---

## 📝 注意事項

1. プライバシー: 収集したデータの取り扱いに注意
2. 長時間使用: 目の疲労に注意し、適度に休憩
3. 実験目的: 本システムは研究・学習目的です
4. データ品質: 良質なデータが高精度の鍵

---

**最終更新**: 2024年
**作成者**: 眠気推定システム開発チーム
