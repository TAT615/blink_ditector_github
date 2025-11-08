# プロジェクト構造

## 📁 ディレクトリ構造

```
drowsiness-estimation-system/
│
├── 📄 コアモジュール（必須）
│   ├── blink_detector.py                    # 既存: 4段階EAR検出
│   ├── blink_feature_extractor.py           # 新規: 特徴量抽出
│   ├── lstm_drowsiness_model.py             # 新規: LSTMモデル
│   ├── drowsiness_data_collector.py         # 新規: データ収集
│   ├── drowsiness_data_manager.py           # 新規: データ管理
│   ├── train_drowsiness_model.py            # 新規: モデル訓練
│   └── realtime_drowsiness_estimator.py     # 新規: リアルタイム推定
│
├── 📄 サポートファイル
│   ├── training_config.json                 # 訓練設定サンプル
│   ├── test_training_script.py              # 動作確認テスト
│   └── load_and_use_model.py               # モデル使用サンプル
│
├── 📚 ドキュメント
│   ├── FINAL_README.md                      # プロジェクト完成まとめ
│   ├── PROJECT_PROGRESS.md                  # 進捗状況
│   ├── DATA_COLLECTION_README.md            # データ収集ガイド
│   ├── MODEL_TRAINING_README.md             # モデル訓練ガイド
│   └── REALTIME_SYSTEM_README.md            # リアルタイムガイド
│
├── 📊 データディレクトリ（実行時に生成）
│   └── drowsiness_training_data/
│       ├── sessions/                        # セッション情報
│       │   ├── normal_YYYYMMDD_HHMMSS_info.json
│       │   ├── normal_YYYYMMDD_HHMMSS_blinks.csv
│       │   ├── drowsy_YYYYMMDD_HHMMSS_info.json
│       │   └── drowsy_YYYYMMDD_HHMMSS_blinks.csv
│       ├── sequences/                       # シーケンスデータ
│       │   ├── normal_YYYYMMDD_HHMMSS_sequences.npz
│       │   └── drowsy_YYYYMMDD_HHMMSS_sequences.npz
│       ├── drowsiness_dataset.npz          # 統合データセット
│       ├── normalization_params.json       # 正規化パラメータ
│       └── statistics.json                  # 統計情報
│
├── 🧠 モデルディレクトリ（訓練時に生成）
│   └── trained_models/
│       ├── drowsiness_lstm_YYYYMMDD_HHMMSS.pth           # 訓練済みモデル
│       ├── drowsiness_lstm_YYYYMMDD_HHMMSS_metadata.json # メタデータ
│       └── logs/
│           ├── drowsiness_lstm_YYYYMMDD_HHMMSS_history.png
│           └── drowsiness_lstm_YYYYMMDD_HHMMSS_confusion_matrix.png
│
├── 📝 ログディレクトリ（推定時に生成）
│   └── drowsiness_logs/
│       └── session_YYYYMMDD_HHMMSS.json
│
└── 📦 その他必要ファイル
    └── shape_predictor_68_face_landmarks.dat # dlib顔ランドマーク
```

---

## 📄 ファイル詳細

### コアモジュール

#### 1. blink_detector.py（既存）
**役割**: 4段階EAR検出  
**機能**:
- 顔検出・ランドマーク検出
- EAR計算
- 4段階状態遷移検出
- 個人キャリブレーション
- 時間パラメータ抽出（t1, t2, t3）

**依存**: dlib, OpenCV

---

#### 2. blink_feature_extractor.py
**役割**: 特徴量抽出  
**機能**:
- 瞬き係数 (To/Tc) 計算
- 6次元特徴ベクトル生成
- データ検証
- 正規化（Z-score）
- シーケンス生成

**入力**: 瞬きデータ (t1, t2, t3, ear_min)  
**出力**: 6次元特徴ベクトル

---

#### 3. lstm_drowsiness_model.py
**役割**: LSTMモデル定義  
**機能**:
- DrowsinessLSTMモデル（2層LSTM）
- BlinkSequenceDatasetクラス
- DrowsinessEstimator（訓練・推論）
- モデル保存・読み込み

**入力**: (batch, 10, 6)  
**出力**: (batch, 2) クラス確率

---

#### 4. drowsiness_data_collector.py
**役割**: データ収集  
**機能**:
- インタラクティブUI
- リアルタイム瞬き検出
- ラベリング（正常/眠気）
- データ保存（CSV/JSON/NumPy）
- 統計記録

**キー操作**:
- [N]: 正常状態収集
- [D]: 眠気状態収集
- [SPACE]: 保存

---

#### 5. drowsiness_data_manager.py
**役割**: データ管理  
**機能**:
- 全セッションデータ読み込み
- 訓練/検証/テスト分割
- 正規化
- データセットエクスポート/インポート

---

#### 6. train_drowsiness_model.py
**役割**: モデル訓練  
**機能**:
- データ自動前処理
- モデル訓練（Early Stopping）
- 訓練履歴可視化
- モデル評価
- コマンドライン引数対応

**使用例**:
```bash
python train_drowsiness_model.py --data-dir drowsiness_training_data
```

---

#### 7. realtime_drowsiness_estimator.py
**役割**: リアルタイム推定  
**機能**:
- カメラ入力
- リアルタイム瞬き検出
- LSTM推論
- アラート機能
- UI表示
- ログ記録

**使用例**:
```bash
python realtime_drowsiness_estimator.py --model trained_models/model.pth
```

---

## 🔄 データフロー

### 1. データ収集フェーズ

```
カメラ
  ↓
blink_detector.py
  ↓ (瞬きデータ)
drowsiness_data_collector.py
  ↓ (特徴量)
drowsiness_training_data/
  ├─ sessions/
  └─ sequences/
```

### 2. データ管理フェーズ

```
drowsiness_training_data/
  ↓
drowsiness_data_manager.py
  ↓ (前処理済み)
drowsiness_dataset.npz
normalization_params.json
```

### 3. 訓練フェーズ

```
drowsiness_dataset.npz
  ↓
train_drowsiness_model.py
  ↓ (訓練)
trained_models/
  ├─ model.pth
  ├─ metadata.json
  └─ logs/
```

### 4. 推定フェーズ

```
カメラ → blink_detector.py
  ↓
realtime_drowsiness_estimator.py
  ├─ model.pth (読み込み)
  ├─ normalization_params.json
  ↓
drowsiness_logs/
```

---

## 💾 データ形式

### セッション情報 (JSON)

```json
{
  "session_name": "normal_20240101_120000",
  "label": 0,
  "start_time": 1704096000.0,
  "end_time": 1704096060.0,
  "blink_count": 25,
  "sequence_count": 15
}
```

### 瞬きデータ (CSV)

```csv
timestamp,label,blink_coefficient,tc,to,interval,ear_min,total_duration,t1,t2,t3
1704096001.5,0,1.15,0.12,0.14,2.5,0.15,0.26,1.0,1.12,1.26
```

### シーケンスデータ (NumPy .npz)

```python
data = np.load('sequences.npz')
sequences = data['sequences']  # shape: (n, 10, 6)
labels = data['labels']        # shape: (n,)
```

### モデルファイル (PyTorch .pth)

```python
checkpoint = torch.load('model.pth')
model_state_dict = checkpoint['model_state_dict']
model_params = checkpoint['model_params']
history = checkpoint['history']
```

---

## 🔧 セットアップ

### 1. 必要なファイルの配置

```bash
# プロジェクトディレクトリを作成
mkdir drowsiness-estimation-system
cd drowsiness-estimation-system

# コアモジュールを配置
# （作成された全ファイルをコピー）

# dlib顔ランドマークモデルをダウンロード
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

### 2. 依存ライブラリのインストール

```bash
pip install numpy opencv-python dlib torch scikit-learn matplotlib --break-system-packages
```

### 3. 実行

```bash
# データ収集
python drowsiness_data_collector.py

# 訓練
python train_drowsiness_model.py --data-dir drowsiness_training_data

# リアルタイム推定
python realtime_drowsiness_estimator.py --model trained_models/model.pth
```

---

## 📊 ファイルサイズ目安

| ファイル/ディレクトリ | サイズ |
|---------------------|--------|
| コアモジュール（7ファイル） | 約100KB |
| ドキュメント（5ファイル） | 約200KB |
| shape_predictor_68_face_landmarks.dat | 約95MB |
| drowsiness_training_data/ | 1-10MB（データ量次第） |
| trained_models/ | 1-5MB |
| drowsiness_logs/ | 数KB〜数MB |

**合計**: 約100-150MB（データ量次第）

---

## 🔗 モジュール間の依存関係

```
realtime_drowsiness_estimator.py
  ├─ blink_detector.py
  ├─ blink_feature_extractor.py
  └─ lstm_drowsiness_model.py

train_drowsiness_model.py
  ├─ drowsiness_data_manager.py
  └─ lstm_drowsiness_model.py

drowsiness_data_collector.py
  ├─ blink_detector.py
  └─ blink_feature_extractor.py

drowsiness_data_manager.py
  └─ （外部依存なし）

lstm_drowsiness_model.py
  └─ （外部依存なし）

blink_feature_extractor.py
  └─ （外部依存なし）

blink_detector.py
  └─ （外部依存なし）
```

---

## 🎯 各段階で生成されるファイル

### データ収集後

```
drowsiness_training_data/
├── sessions/
│   ├── normal_*.json (×10以上)
│   ├── normal_*.csv (×10以上)
│   ├── drowsy_*.json (×10以上)
│   └── drowsy_*.csv (×10以上)
├── sequences/
│   ├── normal_*.npz (×10以上)
│   └── drowsy_*.npz (×10以上)
└── statistics.json
```

### データ管理後

```
drowsiness_training_data/
├── drowsiness_dataset.npz
└── normalization_params.json
```

### 訓練後

```
trained_models/
├── drowsiness_lstm_YYYYMMDD_HHMMSS.pth
├── drowsiness_lstm_YYYYMMDD_HHMMSS_metadata.json
└── logs/
    ├── drowsiness_lstm_YYYYMMDD_HHMMSS_history.png
    └── drowsiness_lstm_YYYYMMDD_HHMMSS_confusion_matrix.png
```

### リアルタイム推定後

```
drowsiness_logs/
└── session_YYYYMMDD_HHMMSS.json (×複数)
```

---

## 🎓 学習パス

### 初心者向け

1. `FINAL_README.md` を読む
2. `DATA_COLLECTION_README.md` でデータ収集
3. `MODEL_TRAINING_README.md` で訓練
4. `REALTIME_SYSTEM_README.md` で実行

### 中級者向け

1. 各モジュールのコードを読む
2. パラメータをカスタマイズ
3. 独自の機能を追加

### 上級者向け

1. モデルアーキテクチャを変更
2. 新しい特徴量を追加
3. マルチモーダル学習に拡張

---

## ✅ チェックリスト

### 初回セットアップ

- [ ] Pythonインストール済み（3.7以上）
- [ ] 必要なライブラリインストール済み
- [ ] Webカメラ接続済み
- [ ] shape_predictor_68_face_landmarks.datダウンロード済み
- [ ] 全モジュールファイル配置済み

### データ収集前

- [ ] カメラ動作確認
- [ ] 照明条件確認
- [ ] 十分な時間確保（1時間程度）

### 訓練前

- [ ] データ収集完了（各クラス10セッション以上）
- [ ] ディスク容量確認（数GB推奨）

### リアルタイム推定前

- [ ] モデル訓練完了
- [ ] モデルファイル存在確認
- [ ] 正規化パラメータ存在確認

---

**ファイル構造の完全な説明を提供しました。これでプロジェクト全体の理解が深まります！** ✨
