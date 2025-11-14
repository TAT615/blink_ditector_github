"""
時系列データ保存対応 - データコレクター

EARと楕円パラメータの時系列データを別々にJSONファイルに保存します。
"""

import json
import os
from datetime import datetime
import numpy as np


class BlinkDataCollectorWithTimeSeries:
    """
    瞬きデータを収集し、時系列データを含むJSONファイルに保存
    """
    
    def __init__(self, output_dir="data/blink_sessions"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # セッション情報
        self.session_data = {
            'session_id': None,
            'label': None,
            'start_time': None,
            'end_time': None,
            'duration': 0.0,
            'fps': 0.0,
            'total_blinks': 0,
            'valid_blinks': 0,
            'invalid_blinks': 0,
            'blinks': [],
            'metadata': {}
        }
        
        self.blink_counter = 0
    
    def start_session(self, label, subject_id=None, notes=None):
        """
        セッション開始
        
        Args:
            label (int): ラベル（0=正常, 1=眠気）
            subject_id (str): 被験者ID
            notes (str): メモ
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_str = "normal" if label == 0 else "drowsy"
        
        self.session_data['session_id'] = f"{timestamp}_{label_str}"
        self.session_data['label'] = label
        self.session_data['start_time'] = datetime.now().isoformat()
        self.session_data['blinks'] = []
        self.blink_counter = 0
        
        self.session_data['metadata'] = {
            'created_at': datetime.now().isoformat(),
            'subject_id': subject_id or 'unknown',
            'notes': notes or '',
            'software_version': '1.0.0'
        }
        
        print(f"\n{'='*60}")
        print(f"セッション開始: {self.session_data['session_id']}")
        print(f"ラベル: {label_str}")
        print(f"{'='*60}\n")
    
    def add_blink(self, blink_info):
        """
        瞬きデータを追加
        
        Args:
            blink_info (dict): 瞬き情報（統計量 + 時系列データ）
        """
        if blink_info is None:
            return
        
        self.blink_counter += 1
        
        # 瞬きデータの構造化
        blink_data = {
            'blink_id': self.blink_counter,
            'timestamp': blink_info.get('t1', 0.0),
            
            # 統計量
            'statistics': {
                'closing_time': blink_info.get('closing_time', 0.0),
                'opening_time': blink_info.get('opening_time', 0.0),
                'blink_coefficient': blink_info.get('blink_coefficient', 0.0),
                'total_duration': blink_info.get('total_duration', 0.0),
                'interval': blink_info.get('interval', 0.0),
                'ear_min': blink_info.get('ear_min', 0.0),
                'ellipse_major_axis_max': blink_info.get('ellipse_major_axis_max', 0.0),
                'ellipse_minor_axis_min': blink_info.get('ellipse_minor_axis_min', 0.0),
                'ellipse_area_min': blink_info.get('ellipse_area_min', 0.0),
                'ellipse_angle_change': blink_info.get('ellipse_angle_change', 0.0),
                'ellipse_eccentricity_max': blink_info.get('ellipse_eccentricity_max', 0.0)
            },
            
            # 時系列データ
            'ear_timeseries': blink_info.get('ear_timeseries', []),
            'ellipse_timeseries': blink_info.get('ellipse_timeseries', [])
        }
        
        self.session_data['blinks'].append(blink_data)
        self.session_data['total_blinks'] += 1
        
        # 有効性チェック（簡易版）
        if self._is_valid_blink(blink_data['statistics']):
            self.session_data['valid_blinks'] += 1
        else:
            self.session_data['invalid_blinks'] += 1
    
    def _is_valid_blink(self, stats):
        """瞬きの有効性チェック（簡易版）"""
        tc = stats['closing_time']
        to = stats['opening_time']
        coef = stats['blink_coefficient']
        
        if not (0.025 <= tc <= 1.0):
            return False
        if not (0.05 <= to <= 0.6):
            return False
        if not (0.5 <= coef <= 8.0):
            return False
        
        return True
    
    def end_session(self):
        """
        セッション終了してJSONファイルに保存
        
        Returns:
            str: 保存したファイルパス
        """
        self.session_data['end_time'] = datetime.now().isoformat()
        
        # 期間計算（簡易版）
        if len(self.session_data['blinks']) > 0:
            first_blink_time = self.session_data['blinks'][0]['timestamp']
            last_blink_time = self.session_data['blinks'][-1]['timestamp']
            self.session_data['duration'] = last_blink_time - first_blink_time
        
        # JSONファイルに保存
        filepath = os.path.join(
            self.output_dir,
            f"{self.session_data['session_id']}.json"
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"セッション終了: {self.session_data['session_id']}")
        print(f"総瞬き数: {self.session_data['total_blinks']}")
        print(f"有効な瞬き: {self.session_data['valid_blinks']}")
        print(f"無効な瞬き: {self.session_data['invalid_blinks']}")
        print(f"保存先: {filepath}")
        print(f"{'='*60}\n")
        
        return filepath
    
    def load_session(self, filepath):
        """
        JSONファイルからセッションデータを読み込み
        
        Args:
            filepath (str): JSONファイルパス
            
        Returns:
            dict: セッションデータ
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def visualize_blink_timeseries(self, blink_data):
        """
        1つの瞬きの時系列データを可視化（テキストベース）
        
        Args:
            blink_data (dict): 瞬きデータ
        """
        print(f"\n{'='*80}")
        print(f"瞬き ID: {blink_data['blink_id']}")
        print(f"タイムスタンプ: {blink_data['timestamp']:.3f}秒")
        print(f"{'='*80}")
        
        # 統計量
        stats = blink_data['statistics']
        print("\n【統計量】")
        print(f"  閉眼時間:       {stats['closing_time']*1000:6.1f} ms")
        print(f"  開眼時間:       {stats['opening_time']*1000:6.1f} ms")
        print(f"  瞬き係数:       {stats['blink_coefficient']:6.2f}")
        print(f"  EAR最小値:      {stats['ear_min']:6.3f}")
        print(f"  楕円長軸（最大）: {stats['ellipse_major_axis_max']:6.1f} px")
        print(f"  楕円短軸（最小）: {stats['ellipse_minor_axis_min']:6.1f} px")
        print(f"  楕円面積（最小）: {stats['ellipse_area_min']:6.1f} px²")
        print(f"  楕円角度変化:     {stats['ellipse_angle_change']:6.1f} °")
        
        # EAR時系列
        print("\n【EAR時系列】")
        print(f"  {'Time (s)':>10} | {'State':^10} | {'EAR':>6}")
        print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*6}")
        for ear_data in blink_data['ear_timeseries']:
            t = ear_data['timestamp'] - blink_data['timestamp']
            print(f"  {t:10.3f} | {ear_data['state']:^10} | {ear_data['ear']:6.3f}")
        
        # 楕円時系列
        print("\n【楕円時系列】")
        print(f"  {'Time (s)':>10} | {'State':^10} | {'長軸':>6} | {'短軸':>6} | {'面積':>8} | {'角度':>6}")
        print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}")
        for ellipse_data in blink_data['ellipse_timeseries']:
            t = ellipse_data['timestamp'] - blink_data['timestamp']
            print(f"  {t:10.3f} | {ellipse_data['state']:^10} | "
                  f"{ellipse_data['major_axis']:6.1f} | "
                  f"{ellipse_data['minor_axis']:6.1f} | "
                  f"{ellipse_data['area']:8.1f} | "
                  f"{ellipse_data['angle']:6.1f}")
        
        print(f"{'='*80}\n")


# ===== 使用例 =====
if __name__ == "__main__":
    # データコレクターの初期化
    collector = BlinkDataCollectorWithTimeSeries(
        output_dir="data/blink_sessions_timeseries"
    )
    
    # セッション開始
    collector.start_session(
        label=0,  # 0=正常
        subject_id="subject_001",
        notes="テストセッション"
    )
    
    # 瞬きデータの追加（サンプル）
    sample_blink = {
        't1': 10.5,
        't2': 10.582,
        't3': 10.677,
        'closing_time': 0.082,
        'opening_time': 0.095,
        'blink_coefficient': 1.16,
        'total_duration': 0.177,
        'interval': 2.3,
        'ear_min': 0.15,
        'ellipse_major_axis_max': 32.5,
        'ellipse_minor_axis_min': 8.2,
        'ellipse_area_min': 210.3,
        'ellipse_angle_change': 3.5,
        'ellipse_eccentricity_max': 0.95,
        
        'ear_timeseries': [
            {'timestamp': 10.500, 'ear': 0.28, 'state': 'OPEN'},
            {'timestamp': 10.533, 'ear': 0.25, 'state': 'CLOSING'},
            {'timestamp': 10.566, 'ear': 0.20, 'state': 'CLOSING'},
            {'timestamp': 10.599, 'ear': 0.18, 'state': 'CLOSED'},
            {'timestamp': 10.632, 'ear': 0.15, 'state': 'CLOSED'},
            {'timestamp': 10.665, 'ear': 0.22, 'state': 'OPENING'},
            {'timestamp': 10.698, 'ear': 0.26, 'state': 'OPENING'},
            {'timestamp': 10.731, 'ear': 0.30, 'state': 'OPEN'}
        ],
        
        'ellipse_timeseries': [
            {'timestamp': 10.500, 'state': 'OPEN', 'major_axis': 32.5, 
             'minor_axis': 12.3, 'area': 315.2, 'angle': 2.1, 'eccentricity': 0.93},
            {'timestamp': 10.533, 'state': 'CLOSING', 'major_axis': 31.2, 
             'minor_axis': 10.1, 'area': 248.5, 'angle': 2.5, 'eccentricity': 0.94},
            {'timestamp': 10.566, 'state': 'CLOSING', 'major_axis': 30.5, 
             'minor_axis': 8.5, 'area': 204.6, 'angle': 3.0, 'eccentricity': 0.95},
            {'timestamp': 10.599, 'state': 'CLOSED', 'major_axis': 30.0, 
             'minor_axis': 8.2, 'area': 193.7, 'angle': 3.5, 'eccentricity': 0.95},
            {'timestamp': 10.665, 'state': 'OPENING', 'major_axis': 30.8, 
             'minor_axis': 9.5, 'area': 230.6, 'angle': 2.8, 'eccentricity': 0.94},
            {'timestamp': 10.698, 'state': 'OPENING', 'major_axis': 31.5, 
             'minor_axis': 10.8, 'area': 268.3, 'angle': 2.3, 'eccentricity': 0.93},
            {'timestamp': 10.731, 'state': 'OPEN', 'major_axis': 32.5, 
             'minor_axis': 12.3, 'area': 315.2, 'angle': 2.0, 'eccentricity': 0.93}
        ]
    }
    
    collector.add_blink(sample_blink)
    
    # セッション終了
    filepath = collector.end_session()
    
    # データの読み込みと可視化
    print("\n" + "="*80)
    print("保存されたデータの確認")
    print("="*80)
    
    session_data = collector.load_session(filepath)
    
    if len(session_data['blinks']) > 0:
        collector.visualize_blink_timeseries(session_data['blinks'][0])
