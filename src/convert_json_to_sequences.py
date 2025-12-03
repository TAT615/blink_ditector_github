"""
JSONデータをLSTM学習用のシーケンスファイル(.npz)に変換（12次元完全対応版）

特徴量構成（12次元）:
    [0] closing_time: 閉眼時間 [秒]
    [1] opening_time: 開眼時間 [秒]
    [2] blink_coefficient: 瞬き係数 (opening_time / closing_time)
    [3] interval: 前回の瞬きからの間隔 [秒]
    [4] total_duration: 総持続時間 [秒]
    [5] upper_radius_max: 上まぶた円の最大半径 [px]
    [6] lower_radius_max: 下まぶた円の最大半径 [px]
    [7] vertical_distance_min: 上下円の最小距離 [px]
    [8] radius_diff_max: 半径差の最大値 [px]
    [9] eye_height_min: 目の高さの最小値 [px]
    [10] eye_width_avg: 目の幅の平均値 [px]
    [11] ear_min: EARの最小値

使い方:
    python convert_json_to_sequences.py --input-dir data/sessions --output-dir data/sequences
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path


class JSONToSequenceConverter:
    """
    JSONファイルをLSTM用シーケンスファイルに変換（12次元完全対応版）
    """
    
    # 特徴量名の定義（順序重要）
    FEATURE_NAMES = [
        'closing_time',           # [0]
        'opening_time',           # [1]
        'blink_coefficient',      # [2]
        'interval',               # [3]
        'total_duration',         # [4]
        'upper_radius_max',       # [5]
        'lower_radius_max',       # [6]
        'vertical_distance_min',  # [7]
        'radius_diff_max',        # [8]
        'eye_height_min',         # [9]
        'eye_width_avg',          # [10]
        'ear_min'                 # [11]
    ]
    
    def __init__(self, sequence_length=10):
        """
        初期化
        
        Args:
            sequence_length (int): シーケンス長（デフォルト: 10）
        """
        self.sequence_length = sequence_length
        
        print("=" * 70)
        print("📦 JSON→シーケンス変換システム（12次元完全対応版）")
        print("=" * 70)
        print(f"シーケンス長: {self.sequence_length}")
        print(f"特徴量次元: {len(self.FEATURE_NAMES)}次元")
        print("\n特徴量構成:")
        for i, name in enumerate(self.FEATURE_NAMES):
            print(f"  [{i:2d}] {name}")
        print("=" * 70)
    
    def extract_features_from_blink(self, blink_data):
        """
        瞬きデータから12次元特徴量を抽出
        
        Args:
            blink_data (dict): 瞬きデータ
            
        Returns:
            list: 12次元特徴量 または None
        """
        try:
            stats = blink_data['statistics']
            
            # 12次元特徴量を抽出
            features = [
                stats.get('closing_time', 0.0),           # [0]
                stats.get('opening_time', 0.0),           # [1]
                stats.get('blink_coefficient', 0.0),      # [2]
                stats.get('interval', 0.0),               # [3]
                stats.get('total_duration', 0.0),         # [4]
                stats.get('upper_radius_max', 0.0),       # [5]
                stats.get('lower_radius_max', 0.0),       # [6]
                stats.get('vertical_distance_min', 0.0),  # [7]
                stats.get('radius_diff_max', 0.0),        # [8]
                stats.get('eye_height_min', 0.0),         # [9]
                stats.get('eye_width_avg', 0.0),          # [10]
                stats.get('ear_min', 0.0)                 # [11]
            ]
            
            return features
            
        except KeyError as e:
            print(f"      ⚠️ キーが見つかりません: {e}")
            return None
        except Exception as e:
            print(f"      ⚠️ エラー: {e}")
            return None
    
    def _is_valid_blink(self, blink_data):
        """
        瞬きデータの有効性をチェック
        
        Args:
            blink_data (dict): 瞬きデータ
            
        Returns:
            bool: 有効な場合True
        """
        try:
            stats = blink_data['statistics']
            
            # 閉眼時間チェック (25ms - 1000ms)
            closing_time = stats['closing_time']
            if not (0.025 <= closing_time <= 1.0):
                return False
            
            # 開眼時間チェック (50ms - 600ms)
            opening_time = stats['opening_time']
            if not (0.05 <= opening_time <= 0.6):
                return False
            
            # 瞬き係数チェック (0.5 - 8.0)
            blink_coefficient = stats['blink_coefficient']
            if not (0.5 <= blink_coefficient <= 8.0):
                return False
            
            return True
            
        except KeyError:
            return False
        except Exception:
            return False
    
    def convert_session_to_sequences(self, json_file):
        """
        1つのJSONファイルをシーケンスに変換
        
        Args:
            json_file (str): JSONファイルパス
            
        Returns:
            tuple: (sequences, labels, session_info) または None
        """
        try:
            # JSONファイル読み込み
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # セッション情報
            session_id = data.get('session_id', os.path.basename(json_file).replace('.json', ''))
            label = data['label']  # 0: 正常, 1: 眠気
            
            # 有効な瞬きのみを抽出
            valid_blinks = []
            skipped_count = 0
            
            for blink in data['blinks']:
                # 有効性チェック
                if self._is_valid_blink(blink):
                    features = self.extract_features_from_blink(blink)
                    if features is not None:
                        valid_blinks.append(features)
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
            
            # シーケンス長に満たない場合はスキップ
            if len(valid_blinks) < self.sequence_length:
                print(f"    ⚠️ スキップ: 有効な瞬きが不足 ({len(valid_blinks)}/{self.sequence_length})")
                return None
            
            # スライディングウィンドウでシーケンスを作成
            sequences = []
            for i in range(len(valid_blinks) - self.sequence_length + 1):
                seq = valid_blinks[i:i + self.sequence_length]
                sequences.append(seq)
            
            # NumPy配列に変換
            sequences = np.array(sequences, dtype=np.float32)
            labels = np.full(len(sequences), label, dtype=np.int64)
            
            # セッション情報
            label_name = "normal" if label == 0 else "drowsy"
            session_info = {
                'session_id': session_id,
                'label': label,
                'label_name': label_name,
                'total_blinks': len(data['blinks']),
                'used_blinks': len(valid_blinks),
                'skipped_blinks': skipped_count,
                'sequence_count': len(sequences)
            }
            
            return sequences, labels, session_info
            
        except json.JSONDecodeError as e:
            print(f"    ❌ JSONパースエラー: {e}")
            return None
        except KeyError as e:
            print(f"    ❌ 必要なキーがありません: {e}")
            return None
        except Exception as e:
            print(f"    ❌ 変換エラー: {e}")
            return None
    
    def convert_directory(self, input_dir, output_dir):
        """
        ディレクトリ内の全JSONファイルを変換
        
        Args:
            input_dir (str): 入力ディレクトリ（JSONファイル）
            output_dir (str): 出力ディレクトリ（NPZファイル）
            
        Returns:
            bool: 成功したかどうか
        """
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # JSONファイル検索
        json_files = list(Path(input_dir).glob('*.json'))
        
        if len(json_files) == 0:
            print(f"\n❌ JSONファイルが見つかりません: {input_dir}")
            return False
        
        print(f"\n📂 {len(json_files)} 個のJSONファイルを処理中...")
        print("=" * 70)
        
        success_count = 0
        total_sequences = 0
        all_sequences = []
        all_labels = []
        
        for json_file in json_files:
            print(f"\n処理中: {json_file.name}")
            result = self.convert_session_to_sequences(str(json_file))
            
            if result is not None:
                sequences, labels, session_info = result
                
                # NPZファイルとして保存
                session_id = session_info['session_id']
                output_file = os.path.join(output_dir, f"{session_id}_sequences.npz")
                
                np.savez(
                    output_file,
                    sequences=sequences,
                    labels=labels,
                    session_name=session_id,
                    session_info=session_info
                )
                
                label_name = session_info['label_name']
                seq_count = session_info['sequence_count']
                used_blinks = session_info['used_blinks']
                skipped = session_info['skipped_blinks']
                
                print(f"  ✓ {session_id}: {seq_count} sequences ({label_name})")
                print(f"      使用瞬き: {used_blinks}, スキップ: {skipped}")
                
                # 全体に追加
                all_sequences.append(sequences)
                all_labels.append(labels)
                
                success_count += 1
                total_sequences += seq_count
        
        print("\n" + "=" * 70)
        print(f"\n✅ 変換完了")
        print(f"   成功: {success_count}/{len(json_files)} セッション")
        print(f"   総シーケンス数: {total_sequences}")
        print(f"   出力先: {output_dir}")
        
        # 統合データセットを作成
        if len(all_sequences) > 0:
            combined_sequences = np.concatenate(all_sequences, axis=0)
            combined_labels = np.concatenate(all_labels, axis=0)
            
            # 統合ファイルを保存
            combined_file = os.path.join(os.path.dirname(output_dir), 'combined_sequences.npz')
            np.savez(
                combined_file,
                sequences=combined_sequences,
                labels=combined_labels
            )
            print(f"\n📦 統合データセットを保存しました: {combined_file}")
            print(f"   シーケンス形状: {combined_sequences.shape}")
            print(f"   ラベル形状: {combined_labels.shape}")
            
            # クラス別統計
            normal_count = np.sum(combined_labels == 0)
            drowsy_count = np.sum(combined_labels == 1)
            print(f"\n📊 クラス分布:")
            print(f"   正常 (0): {normal_count} ({normal_count/len(combined_labels)*100:.1f}%)")
            print(f"   眠気 (1): {drowsy_count} ({drowsy_count/len(combined_labels)*100:.1f}%)")
            
            # 特徴量の統計情報を表示
            print(f"\n📈 特徴量の統計情報:")
            for i, name in enumerate(self.FEATURE_NAMES):
                values = combined_sequences[:, :, i].flatten()
                print(f"   [{i:2d}] {name:25s}: "
                      f"mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
                      f"min={np.min(values):.4f}, max={np.max(values):.4f}")
        
        return success_count > 0


def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(
        description='JSONデータをLSTM学習用シーケンスファイルに変換（12次元完全対応版）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input-dir', type=str, default='data/sessions',
                       help='入力ディレクトリ（JSONファイル）')
    parser.add_argument('--output-dir', type=str, default='data/sequences',
                       help='出力ディレクトリ（NPZファイル）')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='シーケンス長')
    
    args = parser.parse_args()
    
    # 変換器作成
    converter = JSONToSequenceConverter(sequence_length=args.sequence_length)
    
    # 変換実行
    success = converter.convert_directory(args.input_dir, args.output_dir)
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 変換が正常に完了しました！")
        print("=" * 70)
        print("\n次のステップ:")
        print("  1. 学習を実行:")
        print("     python -m src.train_drowsiness_model --data-dir data")
        print("=" * 70)
        return 0
    else:
        print("\n❌ 変換に失敗しました")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
