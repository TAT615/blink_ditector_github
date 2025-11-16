"""
JSONデータをLSTM学習用のシーケンスファイル(.npz)に変換（修正版）

使い方:
    python convert_json_to_sequences_v2.py --input-dir data/sessions --output-dir data/sequences
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path


class JSONToSequenceConverter:
    """
    JSONファイルをLSTM用シーケンスファイルに変換（修正版）
    """
    
    def __init__(self, sequence_length=10):
        """
        初期化
        
        Args:
            sequence_length (int): シーケンス長（デフォルト: 10）
        """
        self.sequence_length = sequence_length
        
        print("=" * 70)
        print("📦 JSON→シーケンス変換システム（修正版）")
        print("=" * 70)
        print(f"シーケンス長: {self.sequence_length}")
    
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
            
            # 基本6次元
            closing_time = stats['closing_time']
            opening_time = stats['opening_time']
            blink_coefficient = stats['blink_coefficient']
            
            # 時刻情報（timestampから計算 or interval/total_durationを使用）
            timestamp = blink_data.get('timestamp', 0.0)
            total_duration = stats.get('total_duration', closing_time + opening_time)
            interval = stats.get('interval', 0.0)
            
            # 2円パラメータ6次元
            c1_center_x = stats.get('c1_center_x', 0.0)
            c1_center_y = stats.get('c1_center_y', 0.0)
            c1_radius = stats.get('c1_radius', 0.0)
            c2_center_x = stats.get('c2_center_x', 0.0)
            c2_center_y = stats.get('c2_center_y', 0.0)
            c2_radius = stats.get('c2_radius', 0.0)
            
            # 12次元特徴量
            # 注: 元の設計では t1, t2, t3 でしたが、実際のJSONには存在しないため
            #     timestamp, total_duration, interval を使用
            features = [
                closing_time,      # 0: 閉眼時間
                opening_time,      # 1: 開眼時間
                blink_coefficient, # 2: 瞬き係数
                timestamp,         # 3: タイムスタンプ
                total_duration,    # 4: 総持続時間
                interval,          # 5: 瞬き間隔
                c1_center_x,       # 6: 上まぶた円 中心X
                c1_center_y,       # 7: 上まぶた円 中心Y
                c1_radius,         # 8: 上まぶた円 半径
                c2_center_x,       # 9: 下まぶた円 中心X
                c2_center_y,       # 10: 下まぶた円 中心Y
                c2_radius          # 11: 下まぶた円 半径
            ]
            
            return features
            
        except KeyError as e:
            print(f"      ⚠️ キーが見つかりません: {e}")
            return None
        except Exception as e:
            print(f"      ⚠️ エラー: {e}")
            return None
    
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
            session_id = data['session_id']
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
            
            if len(valid_blinks) < self.sequence_length:
                print(f"  ⚠️ {session_id}: 瞬き数不足 ({len(valid_blinks)} < {self.sequence_length})")
                return None
            
            # シーケンス生成（スライディングウィンドウ）
            sequences = []
            labels = []
            
            for i in range(len(valid_blinks) - self.sequence_length + 1):
                sequence = valid_blinks[i:i + self.sequence_length]
                sequences.append(sequence)
                labels.append(label)
            
            sequences = np.array(sequences, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            
            # セッション情報
            session_info = {
                'session_id': session_id,
                'user_id': data.get('user_id', 'unknown'),
                'label': label,
                'label_name': 'normal' if label == 0 else 'drowsy',
                'kss_score': data.get('kss_score', 0),
                'total_blinks': data['total_blinks'],
                'valid_blinks': data.get('valid_blinks', len(valid_blinks)),
                'used_blinks': len(valid_blinks),
                'skipped_blinks': skipped_count,
                'sequence_count': len(sequences)
            }
            
            return sequences, labels, session_info
            
        except Exception as e:
            print(f"  ❌ {json_file}: エラー - {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _is_valid_blink(self, blink_data):
        """
        瞬きの有効性をチェック
        
        Args:
            blink_data (dict): 瞬きデータ
            
        Returns:
            bool: 有効かどうか
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
        
        for json_file in json_files:
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
                
                success_count += 1
                total_sequences += seq_count
        
        print("=" * 70)
        print(f"\n✅ 変換完了")
        print(f"   成功: {success_count}/{len(json_files)} セッション")
        print(f"   総シーケンス数: {total_sequences}")
        print(f"   出力先: {output_dir}")
        
        return success_count > 0


def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(
        description='JSONデータをLSTM学習用シーケンスファイルに変換（修正版）',
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
        print("  1. src/train_drowsiness_model.py の438行目を修正:")
        print("     'input_size': 6, → 'input_size': 12,")
        print("\n  2. 学習を実行:")
        print("     python -m src.train_drowsiness_model --data-dir data")
        print("=" * 70)
        return 0
    else:
        print("\n❌ 変換に失敗しました")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())