"""
LSTM眠気推定モデル
LSTM-based Drowsiness Estimation Model

論文で提案されたLSTMアーキテクチャに基づく眠気推定モデル
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime


class DrowsinessLSTM(nn.Module):
    """
    眠気推定用LSTMモデル
    
    アーキテクチャ:
    - 入力: (batch_size, sequence_length=10, features=6)
    - LSTM層1: 64ユニット + Dropout(0.3)
    - LSTM層2: 32ユニット + Dropout(0.3)
    - 全結合層: 32ユニット + ReLU
    - 出力層: 2クラス (正常/眠気) + Softmax
    """
    
    def __init__(self, input_size=6, hidden_size1=64, hidden_size2=32, 
                 fc_size=32, num_classes=2, dropout_rate=0.3):
        """
        初期化
        
        Args:
            input_size (int): 入力特徴量の次元数（デフォルト: 6）
            hidden_size1 (int): LSTM第1層のユニット数
            hidden_size2 (int): LSTM第2層のユニット数
            fc_size (int): 全結合層のユニット数
            num_classes (int): 出力クラス数（正常/眠気 = 2）
            dropout_rate (float): ドロップアウト率
        """
        super(DrowsinessLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.fc_size = fc_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # LSTM層1
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            num_layers=1,
            batch_first=True,
            dropout=0  # 単層なのでここでは0
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # LSTM層2
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size2,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 全結合層
        self.fc1 = nn.Linear(hidden_size2, fc_size)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # 出力層
        self.fc2 = nn.Linear(fc_size, num_classes)
        
        # 初期化
        self._init_weights()
    
    def _init_weights(self):
        """重みの初期化"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x):
        """
        順伝播
        
        Args:
            x (torch.Tensor): 入力データ (batch_size, sequence_length, input_size)
        
        Returns:
            torch.Tensor: 出力 (batch_size, num_classes)
        """
        # LSTM層1
        out, (h1, c1) = self.lstm1(x)
        out = self.dropout1(out)
        
        # LSTM層2
        out, (h2, c2) = self.lstm2(out)
        out = self.dropout2(out)
        
        # 最後のタイムステップの出力を取得
        out = out[:, -1, :]
        
        # 全結合層
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        # 出力層
        out = self.fc2(out)
        
        return out
    
    def predict_proba(self, x):
        """
        クラス確率を予測
        
        Args:
            x (torch.Tensor): 入力データ
        
        Returns:
            torch.Tensor: クラス確率 (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs
    
    def predict(self, x):
        """
        クラスを予測
        
        Args:
            x (torch.Tensor): 入力データ
        
        Returns:
            torch.Tensor: 予測クラス (batch_size,)
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


class BlinkSequenceDataset(Dataset):
    """
    瞬きシーケンスデータセット
    """
    
    def __init__(self, sequences, labels):
        """
        初期化
        
        Args:
            sequences (np.ndarray): シーケンスデータ (n_samples, sequence_length, features)
            labels (np.ndarray): ラベル (n_samples,) 0: 正常, 1: 眠気
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class DrowsinessEstimator:
    """
    眠気推定システムのトレーナー・推論器
    """
    
    def __init__(self, model_params=None, device=None):
        """
        初期化
        
        Args:
            model_params (dict): モデルパラメータ
            device (str): デバイス ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # デフォルトパラメータ
        default_params = {
            'input_size': 6,
            'hidden_size1': 64,
            'hidden_size2': 32,
            'fc_size': 32,
            'num_classes': 2,
            'dropout_rate': 0.3
        }
        
        if model_params is not None:
            default_params.update(model_params)
        
        self.model_params = default_params
        self.model = DrowsinessLSTM(**default_params).to(self.device)
        
        # 訓練履歴
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Early Stopping用
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
        print(f"🧠 モデル初期化完了")
        print(f"   デバイス: {self.device}")
        print(f"   パラメータ数: {self._count_parameters():,}")
    
    def _count_parameters(self):
        """パラメータ数をカウント"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_model(self, train_sequences, train_labels, 
                   val_sequences=None, val_labels=None,
                   epochs=100, batch_size=32, learning_rate=0.001,
                   patience=10, verbose=True):
        """
        モデルを訓練
        
        Args:
            train_sequences (np.ndarray): 訓練用シーケンス
            train_labels (np.ndarray): 訓練用ラベル
            val_sequences (np.ndarray): 検証用シーケンス
            val_labels (np.ndarray): 検証用ラベル
            epochs (int): エポック数
            batch_size (int): バッチサイズ
            learning_rate (float): 学習率
            patience (int): Early Stoppingの忍耐値
            verbose (bool): 詳細表示
        
        Returns:
            dict: 訓練履歴
        """
        # データセット作成
        train_dataset = BlinkSequenceDataset(train_sequences, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_sequences is not None and val_labels is not None:
            val_dataset = BlinkSequenceDataset(val_sequences, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            use_validation = True
        else:
            use_validation = False
        
        # 損失関数と最適化手法
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"\n🚀 訓練開始")
        print(f"   エポック数: {epochs}")
        print(f"   バッチサイズ: {batch_size}")
        print(f"   学習率: {learning_rate}")
        print(f"   訓練データ数: {len(train_sequences)}")
        if use_validation:
            print(f"   検証データ数: {len(val_sequences)}")
        print("=" * 70)
        
        # 訓練ループ
        for epoch in range(epochs):
            # 訓練フェーズ
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_sequences, batch_labels in train_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # 順伝播
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_labels)
                
                # 逆伝播
                loss.backward()
                optimizer.step()
                
                # 統計
                train_loss += loss.item() * batch_sequences.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # エポックごとの平均
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = 100.0 * train_correct / train_total
            
            self.history['train_loss'].append(epoch_train_loss)
            self.history['train_acc'].append(epoch_train_acc)
            
            # 検証フェーズ
            if use_validation:
                val_loss, val_acc = self._validate(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Early Stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1:3d}/{epochs}] "
                          f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
                          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
                
                # Early Stoppingチェック
                if self.patience_counter >= patience:
                    print(f"\n⏹️  Early Stopping at epoch {epoch+1}")
                    print(f"   Best validation loss: {self.best_val_loss:.4f}")
                    # ベストモデルを復元
                    self.model.load_state_dict(self.best_model_state)
                    break
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1:3d}/{epochs}] "
                          f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}%")
        
        print("=" * 70)
        print("✅ 訓練完了")
        
        return self.history
    
    def _validate(self, val_loader, criterion):
        """
        検証を実行
        
        Args:
            val_loader: 検証データローダー
            criterion: 損失関数
        
        Returns:
            tuple: (検証損失, 検証精度)
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_sequences, batch_labels in val_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item() * batch_sequences.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        
        return epoch_val_loss, epoch_val_acc
    
    def predict(self, sequences):
        """
        眠気状態を予測
        
        Args:
            sequences (np.ndarray): 入力シーケンス
        
        Returns:
            np.ndarray: 予測クラス (0: 正常, 1: 眠気)
        """
        self.model.eval()
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        with torch.no_grad():
            predictions = self.model.predict(sequences_tensor)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, sequences):
        """
        眠気状態の確率を予測
        
        Args:
            sequences (np.ndarray): 入力シーケンス
        
        Returns:
            np.ndarray: クラス確率 (n_samples, 2)
        """
        self.model.eval()
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        with torch.no_grad():
            probabilities = self.model.predict_proba(sequences_tensor)
        
        return probabilities.cpu().numpy()
    
    def evaluate(self, test_sequences, test_labels):
        """
        テストデータで評価
        
        Args:
            test_sequences (np.ndarray): テスト用シーケンス
            test_labels (np.ndarray): テスト用ラベル
        
        Returns:
            dict: 評価結果
        """
        predictions = self.predict(test_sequences)
        accuracy = 100.0 * np.mean(predictions == test_labels)
        
        # 混同行列
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(test_labels, predictions)
        report = classification_report(test_labels, predictions, 
                                       target_names=['正常', '眠気'],
                                       output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        print(f"\n📊 評価結果:")
        print(f"   正解率: {accuracy:.2f}%")
        print(f"\n混同行列:")
        print(f"              予測: 正常  眠気")
        print(f"   実際: 正常     {cm[0, 0]:5d}  {cm[0, 1]:5d}")
        print(f"         眠気     {cm[1, 0]:5d}  {cm[1, 1]:5d}")
        
        return results
    
    def save_model(self, filepath, include_history=True):
        """
        モデルを保存
        
        Args:
            filepath (str): 保存先パス
            include_history (bool): 訓練履歴も保存するか
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_params': self.model_params,
            'device': str(self.device)
        }
        
        if include_history:
            save_dict['history'] = self.history
        
        torch.save(save_dict, filepath)
        print(f"✅ モデルを保存しました: {filepath}")
    
    def load_model(self, filepath):
        """
        モデルを読み込み
        
        Args:
            filepath (str): 読み込むファイルパス
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model_params = checkpoint['model_params']
        self.model = DrowsinessLSTM(**self.model_params).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"✅ モデルを読み込みました: {filepath}")
    
    def get_model_summary(self):
        """モデルの概要を表示"""
        print("\n" + "=" * 70)
        print("📋 モデル概要")
        print("=" * 70)
        print(self.model)
        print("=" * 70)
        print(f"総パラメータ数: {self._count_parameters():,}")
        print("=" * 70)


# テスト用コード
if __name__ == "__main__":
    print("=" * 70)
    print("LSTM眠気推定モデルのテスト")
    print("=" * 70)
    
    # ダミーデータの生成
    np.random.seed(42)
    n_samples = 200
    sequence_length = 10
    n_features = 6
    
    # 正常状態のデータ（瞬き係数が高め）
    normal_sequences = np.random.randn(n_samples // 2, sequence_length, n_features).astype(np.float32)
    normal_sequences[:, :, 0] = np.abs(normal_sequences[:, :, 0]) + 1.2  # 瞬き係数
    normal_labels = np.zeros(n_samples // 2, dtype=np.int64)
    
    # 眠気状態のデータ（瞬き係数が低め、時間が長め）
    drowsy_sequences = np.random.randn(n_samples // 2, sequence_length, n_features).astype(np.float32)
    drowsy_sequences[:, :, 0] = np.abs(drowsy_sequences[:, :, 0]) + 0.6  # 瞬き係数（低め）
    drowsy_sequences[:, :, 1:3] = np.abs(drowsy_sequences[:, :, 1:3]) + 0.5  # 時間（長め）
    drowsy_labels = np.ones(n_samples // 2, dtype=np.int64)
    
    # データの結合とシャッフル
    all_sequences = np.vstack([normal_sequences, drowsy_sequences])
    all_labels = np.hstack([normal_labels, drowsy_labels])
    
    # シャッフル
    indices = np.random.permutation(n_samples)
    all_sequences = all_sequences[indices]
    all_labels = all_labels[indices]
    
    # 訓練/検証/テスト分割
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_sequences = all_sequences[:train_size]
    train_labels = all_labels[:train_size]
    val_sequences = all_sequences[train_size:train_size + val_size]
    val_labels = all_labels[train_size:train_size + val_size]
    test_sequences = all_sequences[train_size + val_size:]
    test_labels = all_labels[train_size + val_size:]
    
    print(f"\n📦 データセット:")
    print(f"   訓練データ: {len(train_sequences)} サンプル")
    print(f"   検証データ: {len(val_sequences)} サンプル")
    print(f"   テストデータ: {len(test_sequences)} サンプル")
    
    # モデル作成
    estimator = DrowsinessEstimator()
    estimator.get_model_summary()
    
    # 訓練
    print("\n" + "=" * 70)
    history = estimator.train_model(
        train_sequences, train_labels,
        val_sequences, val_labels,
        epochs=30,
        batch_size=16,
        learning_rate=0.001,
        patience=10,
        verbose=True
    )
    
    # 評価
    print("\n" + "=" * 70)
    results = estimator.evaluate(test_sequences, test_labels)
    
    # モデル保存のテスト
    print("\n" + "=" * 70)
    test_model_path = 'drowsiness_lstm_test.pth'
    estimator.save_model(test_model_path)
    
    # 予測のテスト
    print("\n" + "=" * 70)
    print("🔮 予測テスト:")
    sample = test_sequences[:1]
    pred_class = estimator.predict(sample)
    pred_proba = estimator.predict_proba(sample)
    
    print(f"   入力形状: {sample.shape}")
    print(f"   予測クラス: {pred_class[0]} ({'正常' if pred_class[0] == 0 else '眠気'})")
    print(f"   確率: 正常={pred_proba[0, 0]:.3f}, 眠気={pred_proba[0, 1]:.3f}")
    
    print("\n" + "=" * 70)
    print("テスト完了 ✅")
    print("=" * 70)
