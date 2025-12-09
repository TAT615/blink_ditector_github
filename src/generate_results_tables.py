#!/usr/bin/env python3
"""
Generate experimental results tables and figures from trained model
Author: TAT
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

# Set matplotlib to use English and improve font rendering
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

class ResultsTableGenerator:
    def __init__(self, confusion_matrix=None):
        """
        Initialize with confusion matrix values
        confusion_matrix: [[TN, FP], [FN, TP]]
        """
        if confusion_matrix is None:
            # From uploaded confusion matrix image:
            # Normal: TN=155, FP=29
            # Drowsy: FN=13, TP=189
            self.tn = 155
            self.fp = 29
            self.fn = 13
            self.tp = 189
        else:
            self.tn = confusion_matrix[0][0]
            self.fp = confusion_matrix[0][1]
            self.fn = confusion_matrix[1][0]
            self.tp = confusion_matrix[1][1]
        
        self.total = self.tn + self.fp + self.fn + self.tp
        
    def calculate_metrics(self):
        """Calculate all performance metrics"""
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = (self.tp + self.tn) / self.total
        
        # Precision (for drowsy class)
        metrics['precision'] = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        
        # Recall (Sensitivity) - for drowsy class
        metrics['recall'] = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        
        # Specificity (for normal class)
        metrics['specificity'] = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0
        
        # F1 Score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0
        
        # False Negative Rate (Miss Rate) - Critical for safety
        metrics['false_negative_rate'] = self.fn / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        
        # False Positive Rate
        metrics['false_positive_rate'] = self.fp / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0
        
        return metrics
    
    def generate_performance_table(self):
        """Generate Table 1: Model Performance Metrics"""
        metrics = self.calculate_metrics()
        
        data = {
            'Metric': [
                'Accuracy',
                'Precision (Drowsy)',
                'Recall (Drowsy)',
                'Specificity (Normal)',
                'F1 Score',
                'False Negative Rate',
                'False Positive Rate'
            ],
            'Value': [
                f"{metrics['accuracy']:.3f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['specificity']:.3f}",
                f"{metrics['f1_score']:.3f}",
                f"{metrics['false_negative_rate']:.3f}",
                f"{metrics['false_positive_rate']:.3f}"
            ],
            'Percentage': [
                f"{metrics['accuracy']*100:.1f}%",
                f"{metrics['precision']*100:.1f}%",
                f"{metrics['recall']*100:.1f}%",
                f"{metrics['specificity']*100:.1f}%",
                f"{metrics['f1_score']*100:.1f}%",
                f"{metrics['false_negative_rate']*100:.1f}%",
                f"{metrics['false_positive_rate']*100:.1f}%"
            ],
            'Note': [
                f'{self.tp + self.tn}/{self.total} samples',
                'Reliability of drowsy prediction',
                'Drowsiness detection rate (Critical)',
                'Normal state detection rate',
                'Harmonic mean of P and R',
                f'{self.fn} misses (Safety critical)',
                f'{self.fp} false alarms'
            ]
        }
        
        df = pd.DataFrame(data)
        return df
    
    def generate_confusion_matrix_table(self):
        """Generate Table 2: Confusion Matrix (Numerical)"""
        data = {
            'True Label': ['Normal', 'Drowsy', 'Total'],
            'Predicted: Normal': [
                f'{self.tn} (TN)',
                f'{self.fn} (FN)',
                f'{self.tn + self.fn}'
            ],
            'Predicted: Drowsy': [
                f'{self.fp} (FP)',
                f'{self.tp} (TP)',
                f'{self.fp + self.tp}'
            ],
            'Total': [
                f'{self.tn + self.fp}',
                f'{self.fn + self.tp}',
                f'{self.total}'
            ]
        }
        
        df = pd.DataFrame(data)
        return df
    
    def generate_opening_time_table(self):
        """Generate Table 3: Eye Opening Time Analysis by Subject"""
        # Data from the thesis
        data = {
            'Subject': ['001', '001', '002', '002', '003', '003', 'Overall Average'],
            'State': ['Normal', 'Drowsy', 'Normal', 'Drowsy', 'Normal', 'Drowsy', '-'],
            'Sample Size': [129, 1280, 815, 159, 908, 1494, '-'],
            'Opening Time (ms)': [207.5, 326.4, 179.7, 409.2, 205.5, 286.6, '-'],
            'Extension Rate': [1.00, 1.57, 1.00, 2.28, 1.00, 1.39, 1.75],
            'Statistical Significance': ['-', 'p < 0.001', '-', 'p < 0.001', '-', 'p < 0.001', '-']
        }
        
        df = pd.DataFrame(data)
        return df
    
    def plot_performance_comparison(self, save_path='performance_comparison.png'):
        """Create a bar chart comparing different metrics"""
        metrics = self.calculate_metrics()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Accuracy', 'Precision', 'Recall\n(Sensitivity)', 'Specificity', 'F1 Score']
        values = [
            metrics['accuracy'] * 100,
            metrics['precision'] * 100,
            metrics['recall'] * 100,
            metrics['specificity'] * 100,
            metrics['f1_score'] * 100
        ]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        bars = ax.bar(metric_names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Evaluation Metrics', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=90, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target (90%)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Performance comparison chart saved: {save_path}")
        return fig
    
    def plot_confusion_matrix_detailed(self, save_path='confusion_matrix_detailed.png'):
        """Create an enhanced confusion matrix visualization"""
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Create matrix
        cm = np.array([[self.tn, self.fp],
                       [self.fn, self.tp]])
        
        # Plot
        im = ax.imshow(cm, cmap='Blues', alpha=0.8)
        
        # Labels
        classes = ['Normal', 'Drowsy']
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, fontsize=12, fontweight='bold')
        ax.set_yticklabels(classes, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix with Percentages', fontsize=14, fontweight='bold', pad=20)
        
        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                value = cm[i, j]
                percentage = value / self.total * 100
                
                # Main text
                text = ax.text(j, i, f'{value}',
                              ha="center", va="center",
                              color="white" if value > self.total/4 else "black",
                              fontsize=24, fontweight='bold')
                
                # Percentage text
                text2 = ax.text(j, i + 0.3, f'({percentage:.1f}%)',
                               ha="center", va="center",
                               color="white" if value > self.total/4 else "black",
                               fontsize=11)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Samples', rotation=270, labelpad=20, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Detailed confusion matrix saved: {save_path}")
        return fig
    
    def plot_recall_calculation_diagram(self, save_path='recall_calculation_diagram.png'):
        """Create a diagram explaining how recall is calculated"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Recall Calculation Method (93.6%)', 
               ha='center', va='top', fontsize=16, fontweight='bold')
        
        # Data flow diagram
        y_start = 0.85
        
        # Test dataset
        ax.text(0.5, y_start, 'Test Dataset: 386 Sequences', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=2))
        
        # Arrow down
        ax.annotate('', xy=(0.5, y_start - 0.08), xytext=(0.5, y_start - 0.02),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Split into two classes
        y_split = y_start - 0.15
        
        # Normal samples
        ax.text(0.25, y_split, 'Actual Normal\n184 sequences', 
               ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='black', linewidth=2))
        
        # Drowsy samples (highlighted)
        ax.text(0.75, y_split, 'Actual Drowsy\n202 sequences', 
               ha='center', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFD700', edgecolor='red', linewidth=3))
        
        # Results for Normal
        y_result = y_split - 0.15
        ax.text(0.15, y_result, f'Correctly identified\nas Normal: {self.tn}', 
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        ax.text(0.35, y_result, f'Misclassified\nas Drowsy: {self.fp}', 
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
        
        # Results for Drowsy (highlighted)
        ax.text(0.65, y_result, f'Missed\n(False Negative): {self.fn}', 
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFB6C6', edgecolor='red', linewidth=2))
        
        ax.text(0.85, y_result, f'Correctly detected\nas Drowsy: {self.tp}', 
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#90EE90', edgecolor='green', linewidth=2))
        
        # Recall calculation
        y_calc = y_result - 0.2
        recall_value = self.tp / (self.tp + self.fn) * 100
        
        ax.text(0.5, y_calc, 
               f'Recall = True Positives / (True Positives + False Negatives)',
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', linewidth=2))
        
        ax.text(0.5, y_calc - 0.08, 
               f'Recall = {self.tp} / ({self.tp} + {self.fn}) = {self.tp} / {self.tp + self.fn} = {recall_value:.1f}%',
               ha='center', va='center', fontsize=13, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFD700', edgecolor='red', linewidth=3))
        
        # Explanation
        y_note = y_calc - 0.18
        ax.text(0.5, y_note,
               'High Recall (93.6%) means the system successfully detects\nmost drowsiness cases, minimizing dangerous misses.',
               ha='center', va='center', fontsize=10, style='italic',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Recall calculation diagram saved: {save_path}")
        return fig
    
    def generate_all_outputs(self, output_dir='/mnt/user-data/outputs'):
        """Generate all tables and figures"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING EXPERIMENTAL RESULTS TABLES AND FIGURES")
        print("="*60 + "\n")
        
        # Table 1: Performance Metrics
        print("📊 Table 1: Model Performance Metrics")
        df_performance = self.generate_performance_table()
        print(df_performance.to_string(index=False))
        df_performance.to_csv(f'{output_dir}/table1_performance_metrics.csv', index=False)
        print(f"✓ Saved to: {output_dir}/table1_performance_metrics.csv\n")
        
        # Table 2: Confusion Matrix
        print("📊 Table 2: Confusion Matrix (Numerical)")
        df_confusion = self.generate_confusion_matrix_table()
        print(df_confusion.to_string(index=False))
        df_confusion.to_csv(f'{output_dir}/table2_confusion_matrix.csv', index=False)
        print(f"✓ Saved to: {output_dir}/table2_confusion_matrix.csv\n")
        
        # Table 3: Opening Time Analysis
        print("📊 Table 3: Eye Opening Time Analysis by Subject")
        df_opening = self.generate_opening_time_table()
        print(df_opening.to_string(index=False))
        df_opening.to_csv(f'{output_dir}/table3_opening_time_analysis.csv', index=False)
        print(f"✓ Saved to: {output_dir}/table3_opening_time_analysis.csv\n")
        
        # Generate figures
        print("📈 Generating figures...")
        self.plot_performance_comparison(f'{output_dir}/fig_performance_comparison.png')
        self.plot_confusion_matrix_detailed(f'{output_dir}/fig_confusion_matrix_detailed.png')
        self.plot_recall_calculation_diagram(f'{output_dir}/fig_recall_calculation.png')
        
        print("\n" + "="*60)
        print("✓ ALL OUTPUTS GENERATED SUCCESSFULLY")
        print("="*60)
        print(f"\nOutput directory: {output_dir}/")
        print("\nGenerated files:")
        print("  Tables (CSV):")
        print("    - table1_performance_metrics.csv")
        print("    - table2_confusion_matrix.csv")
        print("    - table3_opening_time_analysis.csv")
        print("  Figures (PNG):")
        print("    - fig_performance_comparison.png")
        print("    - fig_confusion_matrix_detailed.png")
        print("    - fig_recall_calculation.png")


def main():
    """Main function"""
    # Initialize with confusion matrix from the uploaded image
    # [[TN, FP], [FN, TP]]
    # Normal: TN=155, FP=29
    # Drowsy: FN=13, TP=189
    
    generator = ResultsTableGenerator()
    generator.generate_all_outputs()


if __name__ == '__main__':
    main()
