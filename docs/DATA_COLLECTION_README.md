# 段階4: データ収集・管理システム

## 作成されたファイル

1. **drowsiness_data_collector.py** - データ収集システム
2. **drowsiness_data_manager.py** - データ管理システム

---

## 📦 drowsiness_data_collector.py

### 機能

- リアルタイム瞬き検出と特徴量抽出
- 正常/眠気状態のラベリング
- CSV/JSON/NumPy形式でのデータ保存
- インタラクティブなデータ収集

### 使用方法

#### 基本的な使い方

```python
from drowsiness_data_collector import DrowsinessDataCollector

# コレクター作成
collector = DrowsinessDataCollector(
    data_dir="drowsiness_training_data",
    sequence_length=10
)

# インタラクティブモードで実行
collector.run_interactive()
```

#### インタラクティブモードの操作

- **[N]** - 正常状態のデータ収集開始
- **[D]** - 眠気状態のデータ収集開始
- **[SPACE]** - 現在のセッション終了・保存
- **[ESC]** - セッション破棄またはプログラム終了
- **[S]** - 統計情報表示

#### プログラマティックな使用

```python
# コレクター作成とカメラ初期化
collector = DrowsinessDataCollector()
collector.initialize_camera(camera_id=0)

# 正常状態のデータ収集開始
collector.start_session(
    label=collector.LABEL_NORMAL,
    session_name="normal_session_1"
)

# データ収集ループ
while True:
    ret, frame = collector.camera.read()
    if not ret:
        break
    
    processed_frame, blink_detected = collector.collect_frame(frame)
    
    # フレーム表示など
    cv2.imshow('Collector', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# セッション終了
collector.stop_session(save=True)
```

### 保存されるデータ

#### ディレクトリ構造

```
drowsiness_training_data/
├── sessions/
│   ├── normal_20240101_120000_info.json      # セッション情報
│   ├── normal_20240101_120000_blinks.csv     # 瞬きデータ
│   ├── drowsy_20240101_130000_info.json
│   └── drowsy_20240101_130000_blinks.csv
├── sequences/
│   ├── normal_20240101_120000_sequences.npz  # シーケンスデータ
│   └── drowsy_20240101_130000_sequences.npz
└── statistics.json                            # 統計情報
```

#### セッション情報 (JSON)

```json
{
  "session_name": "normal_20240101_120000",
  "label": 0,
  "label_name": "normal",
  "start_time": 1704096000.0,
  "end_time": 1704096060.0,
  "duration": 60.0,
  "blink_count": 25,
  "sequence_count": 15,
  "data_points": 25
}
```

#### 瞬きデータ (CSV)

| timestamp | label | blink_coefficient | tc | to | interval | ear_min | total_duration | t1 | t2 | t3 |
|-----------|-------|-------------------|----|----|----------|---------|----------------|----|----|-----|
| 1704096001.5 | 0 | 1.15 | 0.12 | 0.14 | 2.5 | 0.15 | 0.26 | 1.0 | 1.12 | 1.26 |

#### シーケンスデータ (NumPy)

```python
data = np.load('sequences/session_sequences.npz')
sequences = data['sequences']  # shape: (n_sequences, 10, 6)
labels = data['labels']        # shape: (n_sequences,)
```

---

## 📊 drowsiness_data_manager.py

### 機能

- 全セッションデータの読み込み
- データの正規化（Z-score / Min-Max）
- 訓練/検証/テスト分割（層化抽出対応）
- データセットのエクスポート/インポート
- 統計情報の表示

### 使用方法

#### 基本的な使い方

```python
from drowsiness_data_manager import DrowsinessDataManager

# マネージャー作成
manager = DrowsinessDataManager(data_dir="drowsiness_training_data")

# データ読み込み
manager.load_all_data(verbose=True)

# データ分割（70% 訓練, 15% 検証, 15% テスト）
manager.split_data(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify=True  # クラス比率を保持
)

# 正規化（Z-score）
manager.normalize_data(method='zscore')

# データ取得
train_sequences, train_labels = manager.get_train_data()
val_sequences, val_labels = manager.get_val_data()
test_sequences, test_labels = manager.get_test_data()

# 統計表示
manager.print_statistics()
```

#### データセットのエクスポート

```python
# 処理済みデータセットを保存
manager.export_dataset('drowsiness_dataset.npz')

# 正規化パラメータを保存
manager.save_normalization_params('normalization_params.json')
```

#### データセットのインポート

```python
# 保存済みデータセットを読み込み
manager = DrowsinessDataManager()
manager.load_dataset('drowsiness_dataset.npz')

# すぐに訓練に使用可能
train_sequences, train_labels = manager.get_train_data()
```

---

## 🔄 データ収集のワークフロー

### 1. データ収集

```bash
# インタラクティブモードで実行
python drowsiness_data_collector.py
```

1. プログラム起動後、**[N]** キーで正常状態のセッション開始
2. 30-60秒程度、リラックスして自然に瞬きする
3. **[SPACE]** キーでセッション終了・保存
4. **[D]** キーで眠気状態のセッション開始
5. 眠気を感じる状態（目を閉じ気味、瞬きが遅いなど）を演技
6. **[SPACE]** キーでセッション終了・保存
7. 複数セッション（各10セッション程度）を収集

### 2. データ確認

```python
from drowsiness_data_manager import DrowsinessDataManager

manager = DrowsinessDataManager()
manager.load_all_data()
manager.print_statistics()
```

### 3. データ前処理

```python
# データ分割と正規化
manager.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
manager.normalize_data(method='zscore')

# エクスポート
manager.export_dataset('drowsiness_dataset.npz')
manager.save_normalization_params('normalization_params.json')
```

---

## 📈 データ品質のポイント

### 良いデータを収集するために

1. **適切な環境**
   - 安定した照明
   - カメラが顔を正面から捉えられる位置
   - 背景がシンプル

2. **正常状態のデータ**
   - リラックスした状態
   - 自然な瞬き
   - 画面を見ている状態

3. **眠気状態のデータ**
   - 意図的に瞬きを遅くする
   - 目を半分閉じ気味にする
   - まぶたの動きを緩慢にする

4. **バランス**
   - 正常と眠気のデータ数を同程度にする
   - 各状態で最低10セッション以上

---

## 🐛 トラブルシューティング

### カメラが開けない

```python
# 別のカメラIDを試す
collector.initialize_camera(camera_id=1)
```

### データが保存されない

- `drowsiness_training_data` ディレクトリの書き込み権限を確認
- セッション中に十分な瞬きがあるか確認（最低10回以上推奨）

### 瞬きが検出されない

- 照明を調整
- カメラとの距離を調整
- キャリブレーションを再実行

---

## 💡 次のステップ

データ収集が完了したら、次は：

1. **段階5: モデル訓練スクリプトの作成**
   - 収集したデータでLSTMモデルを訓練

2. **段階6: リアルタイム推定システムの統合**
   - 全てのモジュールを統合してリアルタイム眠気推定システムを構築

---

## 📝 注意事項

- データ収集中は顔がカメラの視野内に常にあるようにする
- 長時間のデータ収集は目の疲労を引き起こす可能性があるため、適度に休憩を取る
- プライバシーに配慮し、収集したデータの取り扱いに注意する
