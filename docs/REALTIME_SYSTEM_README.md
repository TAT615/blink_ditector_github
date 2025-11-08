# 段階6: リアルタイム眠気推定システム

## 🎉 完成！

**全てのモジュールが統合され、リアルタイム眠気推定システムが完成しました！**

---

## 作成されたファイル

**realtime_drowsiness_estimator.py** - リアルタイム推定システム

---

## 🚀 機能

### コア機能
- ✅ カメラからのリアルタイム映像入力
- ✅ 4段階EAR検出による瞬き検出
- ✅ 瞬き係数を含む6次元特徴量抽出
- ✅ LSTMによる眠気推定（過去10回の瞬きから判定）
- ✅ リアルタイム結果表示

### アラート機能
- ✅ 眠気確率が閾値を超えると警告
- ✅ 画面フラッシュアラート（赤い点滅）
- ✅ テキストアラート表示
- ✅ 連続検出による誤検出防止

### 可視化機能
- ✅ リアルタイムEAR値表示
- ✅ 眠気確率表示
- ✅ 状態表示（正常/眠気）
- ✅ 瞬き検出表示
- ✅ 眠気確率グラフ（履歴100回分）
- ✅ 統計情報表示

### ログ機能
- ✅ セッション情報の自動保存
- ✅ 推定履歴の記録
- ✅ 統計情報の記録

---

## 📖 使用方法

### 基本的な使い方

```bash
# 必須: 訓練済みモデルのパスを指定
python realtime_drowsiness_estimator.py \
    --model trained_models/drowsiness_lstm_20240101_120000.pth

# 推奨: 正規化パラメータも指定
python realtime_drowsiness_estimator.py \
    --model trained_models/drowsiness_lstm_20240101_120000.pth \
    --norm-params drowsiness_training_data/normalization_params.json
```

### カスタム設定

```bash
# カメラIDを指定（複数カメラがある場合）
python realtime_drowsiness_estimator.py \
    --model trained_models/drowsiness_lstm_20240101_120000.pth \
    --camera 1

# アラート閾値を調整（0.0-1.0）
python realtime_drowsiness_estimator.py \
    --model trained_models/drowsiness_lstm_20240101_120000.pth \
    --threshold 0.8

# シーケンス長を変更
python realtime_drowsiness_estimator.py \
    --model trained_models/drowsiness_lstm_20240101_120000.pth \
    --sequence-length 15
```

### コマンドライン引数

| 引数 | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `--model` | ✅ | - | 訓練済みモデルのパス (.pth) |
| `--norm-params` | - | 自動検出 | 正規化パラメータのパス (.json) |
| `--camera` | - | `0` | カメラID |
| `--sequence-length` | - | `10` | シーケンス長 |
| `--threshold` | - | `0.7` | アラート閾値 (0.0-1.0) |

---

## 🎮 操作方法

### キーボード操作

| キー | 機能 |
|------|------|
| **SPACE** | 統計情報を表示 |
| **R** | 統計情報をリセット |
| **G** | グラフ表示のON/OFF |
| **S** | セッション情報を保存 |
| **ESC** | システムを終了 |

---

## 📊 画面表示

### メイン表示

```
┌─────────────────────────────────────┐
│ Status: 正常/眠気                   │
│ Drowsy Prob: 45.2%                  │
│ EAR: 0.285                          │
│ BLINK! (瞬き検出時)                │
│ Blinks: 25                          │
│ Predictions: 15                     │
└─────────────────────────────────────┘
```

### アラート表示

眠気が検出されると：
- 🔴 画面が赤く点滅
- ⚠️ "DROWSINESS ALERT" の大きな警告表示
- 🔊 コンソールに警告メッセージ

### グラフ表示

画面右下に眠気確率の時系列グラフ：
- 緑線: 正常範囲
- 赤線: 閾値超過
- 青い点線: アラート閾値

---

## 💾 ログファイル

### ディレクトリ構造

```
drowsiness_logs/
└── session_20240101_120000.json
```

### ログ内容

```json
{
  "session_id": "20240101_120000",
  "timestamp": "2024-01-01T12:00:00",
  "model_path": "trained_models/drowsiness_lstm_20240101_120000.pth",
  "statistics": {
    "session_duration": 300.5,
    "total_frames": 9015,
    "total_blinks": 45,
    "total_predictions": 35,
    "normal_predictions": 28,
    "drowsy_predictions": 7,
    "alert_count": 2
  },
  "prediction_history": [0, 0, 1, 0, ...],
  "drowsy_probability_history": [0.12, 0.25, 0.85, 0.15, ...]
}
```

---

## 🔄 動作フロー

```
1. システム起動
   ↓
2. カメラ初期化
   ↓
3. 個人キャリブレーション (5秒)
   ↓
4. リアルタイム処理ループ開始
   │
   ├─ フレーム取得
   ├─ 瞬き検出
   ├─ 特徴量抽出
   ├─ シーケンス生成（10個溜まったら）
   ├─ LSTM推論
   ├─ 状態更新
   ├─ アラートチェック
   └─ UI描画
   │
   └─ ESCで終了
   ↓
5. 統計表示とログ保存
   ↓
6. クリーンアップ
```

---

## 🎯 アラートロジック

### アラート発動条件

以下の**全て**を満たす場合にアラート発動：

1. **眠気と判定** - LSTMが眠気状態と予測
2. **高い確率** - 眠気確率が閾値以上（デフォルト70%）
3. **連続検出** - 連続3回以上眠気と判定

### アラート解除条件

- 正常状態に戻る
- 眠気確率が閾値を下回る

### 誤検出防止

- 連続検出による確認
- 閾値による信頼度チェック
- 時系列情報の考慮（LSTM）

---

## 📈 統計情報

### セッション統計

```
📊 セッション統計
======================================================================
セッション時間: 300.5秒
総フレーム数: 9015
総瞬き数: 45
総推定回数: 35
  正常: 28 (80.0%)
  眠気: 7 (20.0%)
アラート回数: 2
======================================================================
```

---

## 🎨 カスタマイズ

### アラート閾値の調整

```bash
# 敏感に検出（閾値を下げる）
python realtime_drowsiness_estimator.py --model <model> --threshold 0.5

# 厳密に検出（閾値を上げる）
python realtime_drowsiness_estimator.py --model <model> --threshold 0.9
```

**推奨値:**
- 敏感な検出: `0.5 - 0.6`
- 標準: `0.7`
- 厳密な検出: `0.8 - 0.9`

### 連続検出回数の変更

コード内の `consecutive_drowsy_threshold` を編集：

```python
self.consecutive_drowsy_threshold = 5  # デフォルト: 3
```

---

## 🔧 トラブルシューティング

### カメラが開けない

```
❌ カメラを開けませんでした
```

**対処法:**
- カメラが他のアプリで使用されていないか確認
- カメラIDを変更: `--camera 1`
- カメラの接続を確認

### モデルが見つからない

```
❌ モデルファイルが見つかりません
```

**対処法:**
- モデルファイルのパスを確認
- 訓練が完了しているか確認
- 絶対パスで指定

### 瞬きが検出されない

**対処法:**
- 照明条件を改善
- カメラとの距離を調整（30-50cm程度）
- 顔が正面を向いているか確認
- キャリブレーションをやり直す（再起動）

### 推定が行われない

**確認事項:**
1. 瞬きが検出されているか（"BLINK!" 表示）
2. 10回以上瞬きがあるか（シーケンス生成に必要）
3. 正規化パラメータが読み込まれているか

### アラートが頻繁に出る

**対処法:**
- 閾値を上げる: `--threshold 0.8`
- 連続検出回数を増やす（コード編集）
- モデルを再訓練（データ品質改善）

---

## 💡 使用のコツ

### 良い検出のために

1. **適切な環境**
   - 安定した照明
   - 正面からカメラに向く
   - カメラとの距離30-50cm

2. **キャリブレーション**
   - リラックスして自然に瞬き
   - 5秒間カメラを見続ける

3. **定期的な確認**
   - SPACEキーで統計確認
   - 誤検出が多い場合は閾値調整

### 実用時の注意

- 長時間使用時は目の疲労に注意
- アラートはあくまで参考（過信しない）
- 実際に眠い時は休憩を取る

---

## 🧪 テスト実行

### ダミーモデルでテスト

まず訓練を実行:

```bash
# データ収集
python drowsiness_data_collector.py

# モデル訓練
python train_drowsiness_model.py --data-dir drowsiness_training_data

# リアルタイム推定
python realtime_drowsiness_estimator.py \
    --model trained_models/drowsiness_lstm_YYYYMMDD_HHMMSS.pth
```

---

## 📝 システム統合フロー

### 完全な実行手順

```bash
# 1. データ収集
python drowsiness_data_collector.py
# → 正常と眠気の両方のデータを収集（各10セッション以上）

# 2. モデル訓練
python train_drowsiness_model.py --data-dir drowsiness_training_data
# → モデルが trained_models/ に保存される

# 3. リアルタイム推定
python realtime_drowsiness_estimator.py \
    --model trained_models/drowsiness_lstm_YYYYMMDD_HHMMSS.pth \
    --norm-params drowsiness_training_data/normalization_params.json
```

---

## 🎓 論文との対応

本システムは論文「瞬き係数とLSTMを用いた眠気推定システムの開発」で提案された手法を実装しています：

| 論文の要素 | 実装 |
|-----------|------|
| 4段階EAR検出 | `blink_detector.py` |
| 瞬き係数 (To/Tc) | `blink_feature_extractor.py` |
| 6次元特徴ベクトル | `blink_feature_extractor.py` |
| LSTM分類モデル | `lstm_drowsiness_model.py` |
| リアルタイム推定 | `realtime_drowsiness_estimator.py` |

---

## 🚀 次のステップ

システムが完成したので、以下を試してみてください：

1. **実際のデータで訓練**
   - 十分なデータを収集
   - 高精度なモデルを訓練

2. **パラメータ最適化**
   - アラート閾値の調整
   - モデルの再訓練

3. **長時間テスト**
   - 実際の使用環境でテスト
   - ログを分析して改善

4. **機能拡張**
   - 音声アラート追加
   - ログの可視化ツール
   - Web UIの作成

---

## 📚 関連ドキュメント

- **データ収集**: `DATA_COLLECTION_README.md`
- **モデル訓練**: `MODEL_TRAINING_README.md`
- **プロジェクト進捗**: `PROJECT_PROGRESS.md`

---

## ✅ 完成！

**おめでとうございます！眠気推定システムが完成しました！**

論文で提案されたシステムが実際に動作する形で実装されました。
リアルタイムで眠気を検出し、アラートを出すことができます。

---

**最終更新**: 2024年  
**ステータス**: ✅ 完成
