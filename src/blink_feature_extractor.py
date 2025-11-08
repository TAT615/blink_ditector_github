"""
瞬き特徴量抽出モジュール
Blink Feature Extractor Module

論文に基づき、瞬き係数（To/Tc）を含む6次元特徴ベクトルを抽出します。
"""

import numpy as np
from collections import deque
import time
from typing import Dict, List, Optional, Tuple
import json


class BlinkFeatureExtractor:
    """
    瞬きデータから特徴量を抽出するクラス
    
    抽出する特徴量（6次元）:
    1. 瞬き係数 (To/Tc)
    2. 閉眼時間 Tc [秒]
    3. 開眼時間 To [秒]
    4. 瞬き間隔 [秒]
    5. EAR最小値
    6. 総瞬き時間 (Tc + To) [秒]
    """
    
    def __init__(self, sequence_length=10, buffer_size=100):
        """
        初期化
        
        Args:
            sequence_length (int): LSTM入力用のシーケンス長（過去何回分の瞬きを使うか）
            buffer_size (int): 特徴量履歴の最大保存数
        """
        self.sequence_length = sequence_length
        self.buffer_size = buffer_size
        
        # 瞬きデータの保存
        self.blink_features = deque(maxlen=buffer_size)
        self.raw_blink_data = deque(maxlen=buffer_size)
        
        # 前回の瞬き時刻（瞬き間隔計算用）
        self.last_blink_time = None
        
        # 正規化パラメータ（訓練データから計算）
        self.normalization_params = {
            'mean': None,
            'std': None,
            'is_fitted': False
        }
        
        # 統計情報
        self.stats = {
            'total_blinks': 0,
            'valid_blinks': 0,
            'invalid_blinks': 0,
            'avg_blink_coefficient': [],
            'avg_closing_time': [],
            'avg_opening_time': []
        }
        
        # 異常値検出用の閾値
        self.validity_thresholds = {
            'min_tc': 0.05,      # 最小閉眼時間 [秒]
            'max_tc': 1.0,       # 最大閉眼時間 [秒]
            'min_to': 0.05,      # 最小開眼時間 [秒]
            'max_to': 1.0,       # 最大開眼時間 [秒]
            'min_interval': 0.1, # 最小瞬き間隔 [秒]
            'max_interval': 30.0,# 最大瞬き間隔 [秒]
            'min_ear': 0.0,      # 最小EAR値
            'max_ear': 0.5,      # 最大EAR値
            'min_coefficient': 0.1,  # 最小瞬き係数
            'max_coefficient': 10.0  # 最大瞬き係数
        }
    
    def extract_features(self, blink_data: Dict) -> Optional[np.ndarray]:
        """
        瞬きデータから6次元特徴ベクトルを抽出
        
        Args:
            blink_data (Dict): 瞬きデータ
                - 't1': 閉じ始め時刻 [秒]
                - 't2': 完全閉眼時刻 [秒]
                - 't3': 開き終わり時刻 [秒]
                - 'ear_min': EAR最小値
                
        Returns:
            np.ndarray: 6次元特徴ベクトル、無効な場合はNone
                [瞬き係数, Tc, To, 瞬き間隔, EAR最小値, 総瞬き時間]
        """
        try:
            # 時間パラメータの抽出
            t1 = blink_data.get('t1')
            t2 = blink_data.get('t2')
            t3 = blink_data.get('t3')
            ear_min = blink_data.get('ear_min', 0.0)
            
            # 必須パラメータのチェック
            if t1 is None or t2 is None or t3 is None:
                print("⚠️ 必須パラメータが不足しています")
                self.stats['invalid_blinks'] += 1
                return None
            
            # 閉眼時間 Tc = T2 - T1
            tc = t2 - t1
            
            # 開眼時間 To = T3 - T2
            to = t3 - t2
            
            # 瞬き間隔の計算
            if self.last_blink_time is not None:
                blink_interval = t1 - self.last_blink_time
            else:
                blink_interval = 0.0  # 初回の瞬き
            
            # 総瞬き時間
            total_duration = tc + to
            
            # 瞬き係数 = To / Tc
            if tc > 0:
                blink_coefficient = to / tc
            else:
                print("⚠️ 閉眼時間が0以下です")
                self.stats['invalid_blinks'] += 1
                return None
            
            # データの妥当性チェック
            if not self._validate_features(tc, to, blink_interval, ear_min, blink_coefficient):
                self.stats['invalid_blinks'] += 1
                return None
            
            # 6次元特徴ベクトルの作成
            features = np.array([
                blink_coefficient,  # 1. 瞬き係数
                tc,                 # 2. 閉眼時間
                to,                 # 3. 開眼時間
                blink_interval,     # 4. 瞬き間隔
                ear_min,            # 5. EAR最小値
                total_duration      # 6. 総瞬き時間
            ], dtype=np.float32)
            
            # 前回の瞬き時刻を更新
            self.last_blink_time = t1
            
            # 特徴量を保存
            self.blink_features.append(features)
            self.raw_blink_data.append(blink_data)
            
            # 統計情報の更新
            self.stats['total_blinks'] += 1
            self.stats['valid_blinks'] += 1
            self.stats['avg_blink_coefficient'].append(blink_coefficient)
            self.stats['avg_closing_time'].append(tc)
            self.stats['avg_opening_time'].append(to)
            
            return features
            
        except Exception as e:
            print(f"❌ 特徴量抽出エラー: {e}")
            self.stats['invalid_blinks'] += 1
            return None
    
    def _validate_features(self, tc: float, to: float, interval: float, 
                          ear_min: float, coefficient: float) -> bool:
        """
        特徴量の妥当性をチェック
        
        Args:
            tc: 閉眼時間
            to: 開眼時間
            interval: 瞬き間隔
            ear_min: EAR最小値
            coefficient: 瞬き係数
            
        Returns:
            bool: 妥当な場合True
        """
        # 閉眼時間のチェック
        if not (self.validity_thresholds['min_tc'] <= tc <= self.validity_thresholds['max_tc']):
            print(f"⚠️ 閉眼時間が範囲外: {tc:.3f}秒")
            return False
        
        # 開眼時間のチェック
        if not (self.validity_thresholds['min_to'] <= to <= self.validity_thresholds['max_to']):
            print(f"⚠️ 開眼時間が範囲外: {to:.3f}秒")
            return False
        
        # 瞬き間隔のチェック（初回を除く）
        if interval > 0:
            if not (self.validity_thresholds['min_interval'] <= interval <= self.validity_thresholds['max_interval']):
                print(f"⚠️ 瞬き間隔が範囲外: {interval:.3f}秒")
                return False
        
        # EAR最小値のチェック
        if not (self.validity_thresholds['min_ear'] <= ear_min <= self.validity_thresholds['max_ear']):
            print(f"⚠️ EAR最小値が範囲外: {ear_min:.3f}")
            return False
        
        # 瞬き係数のチェック
        if not (self.validity_thresholds['min_coefficient'] <= coefficient <= self.validity_thresholds['max_coefficient']):
            print(f"⚠️ 瞬き係数が範囲外: {coefficient:.3f}")
            return False
        
        return True
    
    def get_sequence(self, normalize: bool = True) -> Optional[np.ndarray]:
        """
        LSTM入力用のシーケンスデータを取得
        
        Args:
            normalize (bool): 正規化を適用するか
            
        Returns:
            np.ndarray: shape (sequence_length, 6) のシーケンスデータ
                        データ不足の場合はNone
        """
        if len(self.blink_features) < self.sequence_length:
            return None
        
        # 最新のsequence_length個の特徴量を取得
        recent_features = list(self.blink_features)[-self.sequence_length:]
        sequence = np.array(recent_features, dtype=np.float32)
        
        # 正規化
        if normalize and self.normalization_params['is_fitted']:
            sequence = self._normalize(sequence)
        
        return sequence
    
    def get_batch_sequences(self, normalize: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        全ての瞬きからシーケンスデータのバッチを生成
        
        Args:
            normalize (bool): 正規化を適用するか
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: 
                - シーケンスデータ (n_sequences, sequence_length, 6)
                - 対応する生データのリスト
        """
        sequences = []
        raw_data_list = []
        
        if len(self.blink_features) < self.sequence_length:
            return np.array([]), []
        
        # スライディングウィンドウでシーケンスを作成
        for i in range(len(self.blink_features) - self.sequence_length + 1):
            seq = np.array(list(self.blink_features)[i:i + self.sequence_length], dtype=np.float32)
            sequences.append(seq)
            raw_data_list.append(list(self.raw_blink_data)[i:i + self.sequence_length])
        
        sequences = np.array(sequences, dtype=np.float32)
        
        # 正規化
        if normalize and self.normalization_params['is_fitted']:
            # バッチ全体を正規化
            original_shape = sequences.shape
            sequences_reshaped = sequences.reshape(-1, 6)
            sequences_normalized = self._normalize(sequences_reshaped)
            sequences = sequences_normalized.reshape(original_shape)
        
        return sequences, raw_data_list
    
    def fit_normalization(self, features: Optional[np.ndarray] = None):
        """
        正規化パラメータを計算（平均0、標準偏差1）
        
        Args:
            features (np.ndarray): 特徴量データ、Noneの場合は保存済みデータを使用
                                   shape: (n_samples, 6)
        """
        if features is None:
            if len(self.blink_features) == 0:
                print("⚠️ 正規化用のデータがありません")
                return
            features = np.array(list(self.blink_features), dtype=np.float32)
        
        # 平均と標準偏差を計算
        self.normalization_params['mean'] = np.mean(features, axis=0)
        self.normalization_params['std'] = np.std(features, axis=0)
        
        # 標準偏差が0の場合は1に設定（ゼロ除算防止）
        self.normalization_params['std'][self.normalization_params['std'] == 0] = 1.0
        
        self.normalization_params['is_fitted'] = True
        
        print("✅ 正規化パラメータを計算しました")
        print(f"   平均: {self.normalization_params['mean']}")
        print(f"   標準偏差: {self.normalization_params['std']}")
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """
        特徴量を正規化
        
        Args:
            features (np.ndarray): 正規化する特徴量
            
        Returns:
            np.ndarray: 正規化された特徴量
        """
        if not self.normalization_params['is_fitted']:
            print("⚠️ 正規化パラメータが未設定です")
            return features
        
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']
        
        return (features - mean) / std
    
    def denormalize(self, features: np.ndarray) -> np.ndarray:
        """
        正規化された特徴量を元に戻す
        
        Args:
            features (np.ndarray): 正規化された特徴量
            
        Returns:
            np.ndarray: 元のスケールの特徴量
        """
        if not self.normalization_params['is_fitted']:
            print("⚠️ 正規化パラメータが未設定です")
            return features
        
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']
        
        return features * std + mean
    
    def get_statistics(self) -> Dict:
        """
        統計情報を取得
        
        Returns:
            Dict: 統計情報
        """
        stats = self.stats.copy()
        
        if len(self.stats['avg_blink_coefficient']) > 0:
            stats['mean_coefficient'] = np.mean(self.stats['avg_blink_coefficient'])
            stats['std_coefficient'] = np.std(self.stats['avg_blink_coefficient'])
        
        if len(self.stats['avg_closing_time']) > 0:
            stats['mean_tc'] = np.mean(self.stats['avg_closing_time'])
            stats['std_tc'] = np.std(self.stats['avg_closing_time'])
        
        if len(self.stats['avg_opening_time']) > 0:
            stats['mean_to'] = np.mean(self.stats['avg_opening_time'])
            stats['std_to'] = np.std(self.stats['avg_opening_time'])
        
        return stats
    
    def save_normalization_params(self, filepath: str):
        """
        正規化パラメータを保存
        
        Args:
            filepath (str): 保存先ファイルパス
        """
        if not self.normalization_params['is_fitted']:
            print("⚠️ 正規化パラメータが未設定です")
            return
        
        params = {
            'mean': self.normalization_params['mean'].tolist(),
            'std': self.normalization_params['std'].tolist(),
            'is_fitted': self.normalization_params['is_fitted']
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"✅ 正規化パラメータを保存しました: {filepath}")
    
    def load_normalization_params(self, filepath: str):
        """
        正規化パラメータを読み込み
        
        Args:
            filepath (str): 読み込むファイルパス
        """
        try:
            with open(filepath, 'r') as f:
                params = json.load(f)
            
            self.normalization_params['mean'] = np.array(params['mean'], dtype=np.float32)
            self.normalization_params['std'] = np.array(params['std'], dtype=np.float32)
            self.normalization_params['is_fitted'] = params['is_fitted']
            
            print(f"✅ 正規化パラメータを読み込みました: {filepath}")
        except Exception as e:
            print(f"❌ 正規化パラメータの読み込みエラー: {e}")
    
    def reset(self):
        """データをリセット"""
        self.blink_features.clear()
        self.raw_blink_data.clear()
        self.last_blink_time = None
        self.stats = {
            'total_blinks': 0,
            'valid_blinks': 0,
            'invalid_blinks': 0,
            'avg_blink_coefficient': [],
            'avg_closing_time': [],
            'avg_opening_time': []
        }
        print("✅ 特徴量抽出器をリセットしました")
    
    def get_latest_features(self) -> Optional[np.ndarray]:
        """
        最新の特徴量を取得
        
        Returns:
            np.ndarray: 最新の6次元特徴ベクトル
        """
        if len(self.blink_features) == 0:
            return None
        return self.blink_features[-1]
    
    def print_feature_info(self, features: np.ndarray):
        """
        特徴量の情報を表示
        
        Args:
            features (np.ndarray): 6次元特徴ベクトル
        """
        feature_names = [
            "瞬き係数 (To/Tc)",
            "閉眼時間 Tc [秒]",
            "開眼時間 To [秒]",
            "瞬き間隔 [秒]",
            "EAR最小値",
            "総瞬き時間 [秒]"
        ]
        
        print("\n📊 特徴量情報:")
        for i, (name, value) in enumerate(zip(feature_names, features)):
            print(f"  {i+1}. {name}: {value:.4f}")


# テスト用コード
if __name__ == "__main__":
    print("=" * 60)
    print("瞬き特徴量抽出器のテスト")
    print("=" * 60)
    
    # インスタンス作成
    extractor = BlinkFeatureExtractor(sequence_length=10)
    
    # サンプル瞬きデータ
    sample_blinks = [
        {'t1': 0.0, 't2': 0.1, 't3': 0.2, 'ear_min': 0.15},
        {'t1': 2.0, 't2': 2.12, 't3': 2.25, 'ear_min': 0.14},
        {'t1': 4.5, 't2': 4.65, 't3': 4.82, 'ear_min': 0.16},
        {'t1': 7.0, 't2': 7.18, 't3': 7.35, 'ear_min': 0.13},
        {'t1': 10.0, 't2': 10.15, 't3': 10.32, 'ear_min': 0.15},
    ]
    
    print("\n🔍 サンプル瞬きデータから特徴量を抽出:")
    for i, blink in enumerate(sample_blinks):
        print(f"\n瞬き {i+1}:")
        features = extractor.extract_features(blink)
        if features is not None:
            extractor.print_feature_info(features)
    
    # 統計情報の表示
    print("\n" + "=" * 60)
    print("📈 統計情報:")
    stats = extractor.get_statistics()
    print(f"  総瞬き数: {stats['total_blinks']}")
    print(f"  有効瞬き数: {stats['valid_blinks']}")
    print(f"  無効瞬き数: {stats['invalid_blinks']}")
    if 'mean_coefficient' in stats:
        print(f"  平均瞬き係数: {stats['mean_coefficient']:.3f} ± {stats['std_coefficient']:.3f}")
    if 'mean_tc' in stats:
        print(f"  平均閉眼時間: {stats['mean_tc']:.3f} ± {stats['std_tc']:.3f} 秒")
    if 'mean_to' in stats:
        print(f"  平均開眼時間: {stats['mean_to']:.3f} ± {stats['std_to']:.3f} 秒")
    
    # 正規化のテスト
    print("\n" + "=" * 60)
    print("🔧 正規化のテスト:")
    extractor.fit_normalization()
    
    # シーケンスデータの取得（データ不足の場合）
    print("\n" + "=" * 60)
    print("📦 シーケンスデータの取得:")
    sequence = extractor.get_sequence(normalize=True)
    if sequence is None:
        print(f"  ⚠️ データ不足（必要: {extractor.sequence_length}個、現在: {len(extractor.blink_features)}個）")
    else:
        print(f"  ✅ シーケンスデータ取得成功")
        print(f"  Shape: {sequence.shape}")
    
    print("\n" + "=" * 60)
    print("テスト完了 ✅")
    print("=" * 60)
