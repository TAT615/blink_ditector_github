"""
眠気推定用データ管理モジュール（セッション単位分割対応版）
Drowsiness Data Manager with Session-based Splitting

収集したデータの読み込み、前処理、分割を行います。
データリークを防ぐため、セッション単位でデータを分割します。
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import glob


class DrowsinessDataManager:
    """
    眠気推定データセットの管理クラス（セッション単位分割対応）
    
    機能:
    - データの読み込み
    - 正規化・前処理
    - セッション単位での訓練/検証/テスト分割（データリーク防止）
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
        
        # セッション単位のデータ
        self.sessions = []  # [{name, sequences, labels, label}, ...]
        
        # 統合データ
        self.all_sequences = []
        self.all_labels = []
        
        # 分割後のデータ
        self.train_sequences = None
        self.train_labels = None
        self.val_sequences = None
        self.val_labels = None
        self.test_sequences = None
        self.test_labels = None
        
        # 分割情報
        self.train_sessions = []
        self.val_sessions = []
        self.test_sessions = []
        
        # 正規化パラメータ
        self.normalization_params = {
            'mean': None,
            'std': None,
            'is_fitted': False
        }
        
        print("=" * 70)
        print("📦 データマネージャー初期化（セッション単位分割対応版）")
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
            
            self.sessions = []
            
            if verbose:
                print(f"\n📂 {len(sequence_files)} 個のセッションを読み込み中...")
            
            for seq_file in sequence_files:
                # シーケンスデータ読み込み
                data = np.load(seq_file)
                sequences = data['sequences']
                labels = data['labels']
                session_name = str(data['session_name'])
                
                # ラベル（セッション全体のラベル）
                session_label = int(labels[0])
                
                # セッション情報を保存
                self.sessions.append({
                    'name': session_name,
                    'sequences': sequences,
                    'labels': labels,
                    'label': session_label,
                    'count': len(sequences)
                })
                
                if verbose:
                    label_name = 'normal' if session_label == 0 else 'drowsy'
                    print(f"  ✓ {session_name}: {len(sequences)} sequences ({label_name})")
            
            # 統計
            total_sequences = sum(s['count'] for s in self.sessions)
            normal_sessions = [s for s in self.sessions if s['label'] == 0]
            drowsy_sessions = [s for s in self.sessions if s['label'] == 1]
            normal_sequences = sum(s['count'] for s in normal_sessions)
            drowsy_sequences = sum(s['count'] for s in drowsy_sessions)
            
            if verbose:
                print(f"\n✅ データ読み込み完了")
                print(f"   総セッション数: {len(self.sessions)}")
                print(f"     正常セッション: {len(normal_sessions)} ({normal_sequences} sequences)")
                print(f"     眠気セッション: {len(drowsy_sessions)} ({drowsy_sequences} sequences)")
                print(f"   総シーケンス数: {total_sequences}")
                if len(self.sessions) > 0:
                    print(f"   シーケンス形状: {self.sessions[0]['sequences'].shape[1:]}")
            
            return True
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def split_data_by_session(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                               random_state=42, verbose=True):
        """
        セッション単位でデータを分割（データリーク防止）
        
        同じセッション内のシーケンスは全て同じセット（訓練/検証/テスト）に配置されます。
        
        Args:
            train_ratio (float): 訓練データの割合
            val_ratio (float): 検証データの割合
            test_ratio (float): テストデータの割合
            random_state (int): 乱数シード
            verbose (bool): 詳細表示
        """
        if len(self.sessions) == 0:
            print("⚠️ データが読み込まれていません")
            return
        
        # 正常/眠気セッションを分離
        normal_sessions = [s for s in self.sessions if s['label'] == 0]
        drowsy_sessions = [s for s in self.sessions if s['label'] == 1]
        
        if verbose:
            print(f"\n📊 セッション単位でデータ分割中...")
            print(f"   正常セッション: {len(normal_sessions)}")
            print(f"   眠気セッション: {len(drowsy_sessions)}")
        
        # 各クラスを個別に分割（層化抽出）
        def split_sessions(sessions, train_r, val_r, test_r, seed):
            if len(sessions) < 3:
                # セッションが少ない場合は全て訓練に
                return sessions, [], []
            
            # まず訓練+検証 vs テストに分割
            train_val, test = train_test_split(
                sessions, 
                test_size=test_r, 
                random_state=seed
            )
            
            # 次に訓練 vs 検証に分割
            if len(train_val) < 2:
                return train_val, [], test
            
            val_ratio_adjusted = val_r / (train_r + val_r)
            train, val = train_test_split(
                train_val, 
                test_size=val_ratio_adjusted, 
                random_state=seed
            )
            
            return train, val, test
        
        # 正常セッションを分割
        normal_train, normal_val, normal_test = split_sessions(
            normal_sessions, train_ratio, val_ratio, test_ratio, random_state
        )
        
        # 眠気セッションを分割
        drowsy_train, drowsy_val, drowsy_test = split_sessions(
            drowsy_sessions, train_ratio, val_ratio, test_ratio, random_state
        )
        
        # セッション情報を保存
        self.train_sessions = normal_train + drowsy_train
        self.val_sessions = normal_val + drowsy_val
        self.test_sessions = normal_test + drowsy_test
        
        # シーケンスを統合
        def merge_sequences(session_list):
            if len(session_list) == 0:
                return np.array([]), np.array([])
            sequences = np.vstack([s['sequences'] for s in session_list])
            labels = np.concatenate([s['labels'] for s in session_list])
            return sequences, labels
        
        self.train_sequences, self.train_labels = merge_sequences(self.train_sessions)
        self.val_sequences, self.val_labels = merge_sequences(self.val_sessions)
        self.test_sequences, self.test_labels = merge_sequences(self.test_sessions)
        
        if verbose:
            print("\n" + "=" * 70)
            print("📊 セッション単位データ分割完了")
            print("=" * 70)
            
            # 訓練セット
            train_normal = sum(1 for s in self.train_sessions if s['label'] == 0)
            train_drowsy = sum(1 for s in self.train_sessions if s['label'] == 1)
            train_normal_seq = sum(s['count'] for s in self.train_sessions if s['label'] == 0)
            train_drowsy_seq = sum(s['count'] for s in self.train_sessions if s['label'] == 1)
            print(f"訓練セット:")
            print(f"  セッション: {len(self.train_sessions)} (正常: {train_normal}, 眠気: {train_drowsy})")
            print(f"  シーケンス: {len(self.train_sequences)} (正常: {train_normal_seq}, 眠気: {train_drowsy_seq})")
            
            # 検証セット
            val_normal = sum(1 for s in self.val_sessions if s['label'] == 0)
            val_drowsy = sum(1 for s in self.val_sessions if s['label'] == 1)
            val_normal_seq = sum(s['count'] for s in self.val_sessions if s['label'] == 0)
            val_drowsy_seq = sum(s['count'] for s in self.val_sessions if s['label'] == 1)
            print(f"検証セット:")
            print(f"  セッション: {len(self.val_sessions)} (正常: {val_normal}, 眠気: {val_drowsy})")
            print(f"  シーケンス: {len(self.val_sequences)} (正常: {val_normal_seq}, 眠気: {val_drowsy_seq})")
            
            # テストセット
            test_normal = sum(1 for s in self.test_sessions if s['label'] == 0)
            test_drowsy = sum(1 for s in self.test_sessions if s['label'] == 1)
            test_normal_seq = sum(s['count'] for s in self.test_sessions if s['label'] == 0)
            test_drowsy_seq = sum(s['count'] for s in self.test_sessions if s['label'] == 1)
            print(f"テストセット:")
            print(f"  セッション: {len(self.test_sessions)} (正常: {test_normal}, 眠気: {test_drowsy})")
            print(f"  シーケンス: {len(self.test_sequences)} (正常: {test_normal_seq}, 眠気: {test_drowsy_seq})")
            
            print("=" * 70)
            
            # セッション名を表示
            print("\n📋 分割されたセッション:")
            print(f"  訓練: {[s['name'] for s in self.train_sessions]}")
            print(f"  検証: {[s['name'] for s in self.val_sessions]}")
            print(f"  テスト: {[s['name'] for s in self.test_sessions]}")
    
    def normalize_data(self, method='zscore', verbose=True):
        """
        データを正規化（訓練データの統計量を使用）
        
        Args:
            method (str): 正規化方法 ('zscore' or 'minmax')
            verbose (bool): 詳細表示
        """
        if self.train_sequences is None or len(self.train_sequences) == 0:
            print("⚠️ 訓練データがありません")
            return
        
        if method == 'zscore':
            # 訓練データから統計量を計算
            train_flat = self.train_sequences.reshape(-1, self.train_sequences.shape[-1])
            
            mean = np.mean(train_flat, axis=0)
            std = np.std(train_flat, axis=0)
            
            # ゼロ除算防止
            std[std == 0] = 1.0
            
            # 正規化パラメータを保存
            self.normalization_params = {
                'mean': mean.tolist(),
                'std': std.tolist(),
                'is_fitted': True
            }
            
            # 正規化を適用
            self.train_sequences = (self.train_sequences - mean) / std
            
            if len(self.val_sequences) > 0:
                self.val_sequences = (self.val_sequences - mean) / std
            
            if len(self.test_sequences) > 0:
                self.test_sequences = (self.test_sequences - mean) / std
            
            if verbose:
                print(f"\n✅ Z-score正規化完了（訓練データの統計量を使用）")
                print(f"   平均: {mean}")
                print(f"   標準偏差: {std}")
        
        elif method == 'minmax':
            # Min-Max正規化
            train_flat = self.train_sequences.reshape(-1, self.train_sequences.shape[-1])
            
            min_val = np.min(train_flat, axis=0)
            max_val = np.max(train_flat, axis=0)
            
            # ゼロ除算防止
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0
            
            self.normalization_params = {
                'min': min_val.tolist(),
                'max': max_val.tolist(),
                'is_fitted': True
            }
            
            self.train_sequences = (self.train_sequences - min_val) / range_val
            
            if len(self.val_sequences) > 0:
                self.val_sequences = (self.val_sequences - min_val) / range_val
            
            if len(self.test_sequences) > 0:
                self.test_sequences = (self.test_sequences - min_val) / range_val
            
            if verbose:
                print(f"\n✅ Min-Max正規化完了")
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """訓練データを取得"""
        return self.train_sequences, self.train_labels
    
    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """検証データを取得"""
        return self.val_sequences, self.val_labels
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """テストデータを取得"""
        return self.test_sequences, self.test_labels
    
    def save_normalization_params(self, output_path: str):
        """
        正規化パラメータをJSONファイルに保存
        
        Args:
            output_path (str): 出力ファイルパス
        """
        with open(output_path, 'w') as f:
            json.dump(self.normalization_params, f, indent=2)
        print(f"✅ 正規化パラメータを保存: {output_path}")
    
    def export_dataset(self, output_path: str):
        """
        データセットをNumPyファイルにエクスポート
        
        Args:
            output_path (str): 出力ファイルパス
        """
        np.savez(
            output_path,
            train_sequences=self.train_sequences,
            train_labels=self.train_labels,
            val_sequences=self.val_sequences,
            val_labels=self.val_labels,
            test_sequences=self.test_sequences,
            test_labels=self.test_labels,
            normalization_params=self.normalization_params,
            train_session_names=[s['name'] for s in self.train_sessions],
            val_session_names=[s['name'] for s in self.val_sessions],
            test_session_names=[s['name'] for s in self.test_sessions]
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
            
        except Exception as e:
            print(f"❌ データセット読み込みエラー: {e}")
    
    def print_statistics(self):
        """統計情報を表示"""
        print("\n" + "=" * 70)
        print("📊 データセット統計")
        print("=" * 70)
        
        if self.train_sequences is not None and len(self.train_sequences) > 0:
            print(f"訓練データ: {len(self.train_sequences)}")
            print(f"  正常: {np.sum(self.train_labels == 0)}")
            print(f"  眠気: {np.sum(self.train_labels == 1)}")
        
        if self.val_sequences is not None and len(self.val_sequences) > 0:
            print(f"検証データ: {len(self.val_sequences)}")
            print(f"  正常: {np.sum(self.val_labels == 0)}")
            print(f"  眠気: {np.sum(self.val_labels == 1)}")
        
        if self.test_sequences is not None and len(self.test_sequences) > 0:
            print(f"テストデータ: {len(self.test_sequences)}")
            print(f"  正常: {np.sum(self.test_labels == 0)}")
            print(f"  眠気: {np.sum(self.test_labels == 1)}")
        
        print("=" * 70)


# テスト用コード
if __name__ == "__main__":
    print("=" * 70)
    print("データマネージャーのテスト（セッション単位分割）")
    print("=" * 70)
    
    # データマネージャー作成
    manager = DrowsinessDataManager(data_dir="data")
    
    # データ読み込み
    print("\nデータ読み込みを試みます...")
    success = manager.load_all_data(verbose=True)
    
    if success:
        # セッション単位でデータ分割
        print("\nセッション単位でデータを分割します...")
        manager.split_data_by_session(
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15
        )
        
        # 正規化
        print("\nデータを正規化します...")
        manager.normalize_data(method='zscore')
        
        # 統計表示
        manager.print_statistics()
        
        print("\n✅ テスト完了")
    else:
        print("\n⚠️ テストデータがありません")
    
    print("=" * 70)
