"""
眠気推定用データ管理モジュール
Drowsiness Data Manager

収集したデータの読み込み、前処理、分割を行います。
"""

import os
import json
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob


class DrowsinessDataManager:
    """
    眠気推定データセットの管理クラス
    
    機能:
    - データの読み込み
    - 正規化・前処理
    - 訓練/検証/テスト分割
    - バッチ生成
    """
    
    def __init__(self, data_dir="drowsiness_training_data"):
        """
        初期化
        
        Args:
            data_dir (str): データディレクトリ
        """
        self.data_dir = data_dir
        self.sessions_dir = os.path.join(data_dir, 'sessions')
        self.sequences_dir = os.path.join(data_dir, 'sequences')
        
        # データ
        self.all_sequences = []
        self.all_labels = []
        self.session_info = []
        
        # 分割後のデータ
        self.train_sequences = None
        self.train_labels = None
        self.val_sequences = None
        self.val_labels = None
        self.test_sequences = None
        self.test_labels = None
        
        # 正規化パラメータ
        self.scaler = None
        self.normalization_params = {
            'mean': None,
            'std': None,
            'is_fitted': False
        }
        
        print("=" * 70)
        print("📦 データマネージャー初期化")
        print("=" * 70)
        print(f"📁 データディレクトリ: {self.data_dir}")
    
    def load_all_data(self, verbose=True) -> bool:
        """
        全てのセッションデータを読み込み
        
        Args:
            verbose (bool): 詳細表示
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            # シーケンスファイルを検索
            sequence_files = glob.glob(os.path.join(self.sequences_dir, "*_sequences.npz"))
            
            if len(sequence_files) == 0:
                print("⚠️ シーケンスデータが見つかりません")
                return False
            
            self.all_sequences = []
            self.all_labels = []
            self.session_info = []
            
            if verbose:
                print(f"\n📂 {len(sequence_files)} 個のセッションを読み込み中...")
            
            for seq_file in sequence_files:
                # シーケンスデータ読み込み
                data = np.load(seq_file)
                sequences = data['sequences']
                labels = data['labels']
                session_name = str(data['session_name'])
                
                # セッション情報読み込み
                info_file = os.path.join(
                    self.sessions_dir,
                    f"{session_name}_info.json"
                )
                
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        session_info = json.load(f)
                    self.session_info.append(session_info)
                
                # データ追加
                self.all_sequences.append(sequences)
                self.all_labels.append(labels)
                
                if verbose:
                    label_name = 'normal' if labels[0] == 0 else 'drowsy'
                    print(f"  ✓ {session_name}: {len(sequences)} sequences ({label_name})")
            
            # 統合
            self.all_sequences = np.vstack(self.all_sequences)
            self.all_labels = np.concatenate(self.all_labels)
            
            if verbose:
                print(f"\n✅ データ読み込み完了")
                print(f"   総シーケンス数: {len(self.all_sequences)}")
                print(f"   正常: {np.sum(self.all_labels == 0)}")
                print(f"   眠気: {np.sum(self.all_labels == 1)}")
                print(f"   シーケンス形状: {self.all_sequences.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def split_data(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                   random_state=42, stratify=True, verbose=True):
        """
        データを訓練/検証/テストに分割
        
        Args:
            train_ratio (float): 訓練データの割合
            val_ratio (float): 検証データの割合
            test_ratio (float): テストデータの割合
            random_state (int): 乱数シード
            stratify (bool): 層化抽出を行うか
            verbose (bool): 詳細表示
        """
        if len(self.all_sequences) == 0:
            print("⚠️ データが読み込まれていません")
            return
        
        # 割合の確認
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "割合の合計は1.0である必要があります"
        
        # 層化抽出用のパラメータ
        stratify_param = self.all_labels if stratify else None
        
        # 訓練 + (検証 + テスト) に分割
        train_val_ratio = val_ratio / (val_ratio + test_ratio)
        
        self.train_sequences, temp_sequences, self.train_labels, temp_labels = train_test_split(
            self.all_sequences,
            self.all_labels,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=stratify_param
        )
        
        # (検証 + テスト) を 検証とテストに分割
        stratify_param_temp = temp_labels if stratify else None
        
        self.val_sequences, self.test_sequences, self.val_labels, self.test_labels = train_test_split(
            temp_sequences,
            temp_labels,
            test_size=(test_ratio / (val_ratio + test_ratio)),
            random_state=random_state,
            stratify=stratify_param_temp
        )
        
        if verbose:
            print("\n" + "=" * 70)
            print("📊 データ分割完了")
            print("=" * 70)
            print(f"訓練データ: {len(self.train_sequences)} ({train_ratio*100:.1f}%)")
            print(f"  正常: {np.sum(self.train_labels == 0)}, 眠気: {np.sum(self.train_labels == 1)}")
            print(f"検証データ: {len(self.val_sequences)} ({val_ratio*100:.1f}%)")
            print(f"  正常: {np.sum(self.val_labels == 0)}, 眠気: {np.sum(self.val_labels == 1)}")
            print(f"テストデータ: {len(self.test_sequences)} ({test_ratio*100:.1f}%)")
            print(f"  正常: {np.sum(self.test_labels == 0)}, 眠気: {np.sum(self.test_labels == 1)}")
            print("=" * 70)
    
    def normalize_data(self, method='zscore', verbose=True):
        """
        データを正規化
        
        Args:
            method (str): 正規化手法 ('zscore' or 'minmax')
            verbose (bool): 詳細表示
        """
        if self.train_sequences is None:
            print("⚠️ データが分割されていません")
            return
        
        # 訓練データで正規化パラメータを計算
        # shape: (n_samples, sequence_length, features) -> (n_samples * sequence_length, features)
        train_reshaped = self.train_sequences.reshape(-1, self.train_sequences.shape[-1])
        
        if method == 'zscore':
            # 平均0、標準偏差1に正規化
            mean = np.mean(train_reshaped, axis=0)
            std = np.std(train_reshaped, axis=0)
            std[std == 0] = 1.0  # ゼロ除算防止
            
            self.normalization_params['mean'] = mean
            self.normalization_params['std'] = std
            self.normalization_params['is_fitted'] = True
            
            # 適用
            self.train_sequences = self._apply_zscore_normalization(self.train_sequences, mean, std)
            self.val_sequences = self._apply_zscore_normalization(self.val_sequences, mean, std)
            self.test_sequences = self._apply_zscore_normalization(self.test_sequences, mean, std)
            
            if verbose:
                print("\n✅ Z-score正規化完了")
                print(f"   平均: {mean}")
                print(f"   標準偏差: {std}")
        
        elif method == 'minmax':
            # 0-1に正規化
            min_val = np.min(train_reshaped, axis=0)
            max_val = np.max(train_reshaped, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0
            
            self.normalization_params['min'] = min_val
            self.normalization_params['max'] = max_val
            self.normalization_params['range'] = range_val
            self.normalization_params['is_fitted'] = True
            
            # 適用
            self.train_sequences = self._apply_minmax_normalization(self.train_sequences, min_val, range_val)
            self.val_sequences = self._apply_minmax_normalization(self.val_sequences, min_val, range_val)
            self.test_sequences = self._apply_minmax_normalization(self.test_sequences, min_val, range_val)
            
            if verbose:
                print("\n✅ Min-Max正規化完了")
        
        else:
            print(f"❌ 未知の正規化手法: {method}")
    
    def _apply_zscore_normalization(self, data, mean, std):
        """Z-score正規化を適用"""
        original_shape = data.shape
        data_reshaped = data.reshape(-1, data.shape[-1])
        normalized = (data_reshaped - mean) / std
        return normalized.reshape(original_shape)
    
    def _apply_minmax_normalization(self, data, min_val, range_val):
        """Min-Max正規化を適用"""
        original_shape = data.shape
        data_reshaped = data.reshape(-1, data.shape[-1])
        normalized = (data_reshaped - min_val) / range_val
        return normalized.reshape(original_shape)
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """訓練データを取得"""
        return self.train_sequences, self.train_labels
    
    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """検証データを取得"""
        return self.val_sequences, self.val_labels
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """テストデータを取得"""
        return self.test_sequences, self.test_labels
    
    def save_normalization_params(self, filepath: str):
        """
        正規化パラメータを保存
        
        Args:
            filepath (str): 保存先パス
        """
        if not self.normalization_params['is_fitted']:
            print("⚠️ 正規化パラメータが未設定です")
            return
        
        # NumPy配列をリストに変換
        params_to_save = {}
        for key, value in self.normalization_params.items():
            if isinstance(value, np.ndarray):
                params_to_save[key] = value.tolist()
            else:
                params_to_save[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(params_to_save, f, indent=2)
        
        print(f"✅ 正規化パラメータを保存: {filepath}")
    
    def load_normalization_params(self, filepath: str):
        """
        正規化パラメータを読み込み
        
        Args:
            filepath (str): 読み込むファイルパス
        """
        try:
            with open(filepath, 'r') as f:
                params = json.load(f)
            
            # リストをNumPy配列に変換
            for key, value in params.items():
                if isinstance(value, list):
                    self.normalization_params[key] = np.array(value, dtype=np.float32)
                else:
                    self.normalization_params[key] = value
            
            print(f"✅ 正規化パラメータを読み込み: {filepath}")
        except Exception as e:
            print(f"❌ 正規化パラメータの読み込みエラー: {e}")
    
    def get_statistics(self) -> Dict:
        """
        データセットの統計情報を取得
        
        Returns:
            Dict: 統計情報
        """
        stats = {
            'total_sequences': len(self.all_sequences) if len(self.all_sequences) > 0 else 0,
            'total_sessions': len(self.session_info)
        }
        
        if len(self.all_sequences) > 0:
            stats['normal_count'] = int(np.sum(self.all_labels == 0))
            stats['drowsy_count'] = int(np.sum(self.all_labels == 1))
            stats['sequence_shape'] = self.all_sequences.shape
            stats['class_balance'] = {
                'normal': stats['normal_count'] / stats['total_sequences'],
                'drowsy': stats['drowsy_count'] / stats['total_sequences']
            }
        
        if self.train_sequences is not None:
            stats['train_count'] = len(self.train_sequences)
            stats['val_count'] = len(self.val_sequences)
            stats['test_count'] = len(self.test_sequences)
        
        return stats
    
    def print_statistics(self):
        """統計情報を表示"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 70)
        print("📊 データセット統計")
        print("=" * 70)
        print(f"総セッション数: {stats['total_sessions']}")
        print(f"総シーケンス数: {stats['total_sequences']}")
        
        if 'normal_count' in stats:
            print(f"  正常: {stats['normal_count']} ({stats['class_balance']['normal']*100:.1f}%)")
            print(f"  眠気: {stats['drowsy_count']} ({stats['class_balance']['drowsy']*100:.1f}%)")
            print(f"シーケンス形状: {stats['sequence_shape']}")
        
        if 'train_count' in stats:
            print(f"\n分割後:")
            print(f"  訓練: {stats['train_count']}")
            print(f"  検証: {stats['val_count']}")
            print(f"  テスト: {stats['test_count']}")
        
        print("=" * 70)
    
    def export_dataset(self, output_path: str):
        """
        データセットを1つのファイルにエクスポート
        
        Args:
            output_path (str): 出力ファイルパス
        """
        if self.train_sequences is None:
            print("⚠️ データが分割されていません")
            return
        
        np.savez(
            output_path,
            train_sequences=self.train_sequences,
            train_labels=self.train_labels,
            val_sequences=self.val_sequences,
            val_labels=self.val_labels,
            test_sequences=self.test_sequences,
            test_labels=self.test_labels,
            normalization_params=self.normalization_params
        )
        
        print(f"✅ データセットをエクスポート: {output_path}")
    
    def load_dataset(self, input_path: str):
        """
        エクスポートされたデータセットを読み込み
        
        Args:
            input_path (str): 入力ファイルパス
        """
        try:
            data = np.load(input_path, allow_pickle=True)
            
            self.train_sequences = data['train_sequences']
            self.train_labels = data['train_labels']
            self.val_sequences = data['val_sequences']
            self.val_labels = data['val_labels']
            self.test_sequences = data['test_sequences']
            self.test_labels = data['test_labels']
            
            if 'normalization_params' in data:
                self.normalization_params = data['normalization_params'].item()
            
            print(f"✅ データセットを読み込み: {input_path}")
            self.print_statistics()
            
        except Exception as e:
            print(f"❌ データセット読み込みエラー: {e}")


# テスト用コード
if __name__ == "__main__":
    print("=" * 70)
    print("データマネージャーのテスト")
    print("=" * 70)
    
    # データマネージャー作成
    manager = DrowsinessDataManager()
    
    # データ読み込み（テストデータがあれば）
    print("\nデータ読み込みを試みます...")
    success = manager.load_all_data(verbose=True)
    
    if success:
        # 統計表示
        manager.print_statistics()
        
        # データ分割
        print("\nデータを分割します...")
        manager.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        # 正規化
        print("\nデータを正規化します...")
        manager.normalize_data(method='zscore')
        
        # 統計表示
        manager.print_statistics()
        
        # 正規化パラメータの保存
        manager.save_normalization_params('normalization_params.json')
        
        # データセットのエクスポート
        manager.export_dataset('drowsiness_dataset.npz')
        
        print("\n✅ テスト完了")
    else:
        print("\n⚠️ テストデータがありません")
        print("   drowsiness_data_collector.py でデータを収集してください")
    
    print("=" * 70)
