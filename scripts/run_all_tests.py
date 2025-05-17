#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skrypt do uruchomienia pełnych testów różnych algorytmów i podejść do wykrywania fraudów.
Wykonuje następujące kroki:
1. Trenuje modele podstawowe i zaawansowane z różnymi algorytmami
2. Testuje adaptacyjne progi dla różnych kombinacji cech
3. Przeprowadza porównanie wszystkich podejść
4. Generuje podsumowanie wyników
"""

import os
import json
import subprocess
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def run_command(command):
    """Uruchamia komendę i zwraca wynik."""
    print(f"\nUruchamianie: {command}\n")
    subprocess.run(command, shell=True, check=True)

def train_all_models(output_dir):
    """Trenuje wszystkie modele z różnymi algorytmami."""
    print("\nTrenowanie modeli...")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nRandom Forest - podstawowe cechy")
    run_command(f"python scripts/enhanced_model.py --model_type randomforest --no_advanced_features --output_dir {output_dir}/randomforest_basic")
    
    print("\nRandom Forest - zaawansowane cechy")
    run_command(f"python scripts/enhanced_model.py --model_type randomforest --output_dir {output_dir}/randomforest_advanced")
    
    try:
        import xgboost
        print("\nXGBoost - podstawowe cechy")
        run_command(f"python scripts/enhanced_model.py --model_type xgboost --no_advanced_features --output_dir {output_dir}/xgboost_basic")
        
        print("\nXGBoost - zaawansowane cechy")
        run_command(f"python scripts/enhanced_model.py --model_type xgboost --output_dir {output_dir}/xgboost_advanced")
    except ImportError:
        print("XGBoost niedostępny")
    
    try:
        import lightgbm
        print("\nLightGBM - podstawowe cechy")
        run_command(f"python scripts/enhanced_model.py --model_type lightgbm --no_advanced_features --output_dir {output_dir}/lightgbm_basic")
        
        print("\nLightGBM - zaawansowane cechy")
        run_command(f"python scripts/enhanced_model.py --model_type lightgbm --output_dir {output_dir}/lightgbm_advanced")
    except ImportError:
        print("LightGBM niedostępny")

def test_adaptive_thresholds(model_dirs, output_dir):
    """Testuje adaptacyjne progi dla wszystkich modeli."""
    print("\nTestowanie adaptacyjnych progów...")
    os.makedirs(output_dir, exist_ok=True)
    
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        model_path = f"{model_dir}/{model_name}_model.joblib"
        
        if not os.path.exists(model_path):
            print(f"Model {model_path} nie istnieje")
            continue
        
        print(f"\nTestowanie {model_name}")
        adaptive_output_dir = f"{output_dir}/{model_name}"
        
        run_command(f"python scripts/test_adaptive_thresholds.py --model_path {model_path} " +
                   f"--output_dir {adaptive_output_dir} --test_size 50000")

def collect_results(model_dirs, adaptive_dirs, output_dir):
    """Zbiera wyniki wszystkich testów i generuje podsumowanie."""
    print("\nZbieranie wyników...")
    summary_dir = f"{output_dir}/summary"
    os.makedirs(summary_dir, exist_ok=True)
    
    model_results = []
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        result_path = f"{model_dir}/{model_name}_results.json"
        
        if not os.path.exists(result_path):
            print(f"Brak wyników dla {model_name}")
            continue
        
        with open(result_path, 'r') as f:
            result = json.load(f)
        result['model_name'] = model_name
        model_results.append(result)
    
    adaptive_results = []
    for adaptive_dir in adaptive_dirs:
        model_name = os.path.basename(adaptive_dir)
        result_path = f"{adaptive_dir}/adaptive_thresholds_results.json"
        
        if not os.path.exists(result_path):
            print(f"Brak wyników adaptacyjnych dla {model_name}")
            continue
        
        with open(result_path, 'r') as f:
            result = json.load(f)
        result['model_name'] = model_name
        adaptive_results.append(result)
    
    if model_results:
        models_df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'Cechy zaawansowane': r['use_advanced_features'],
                'Najlepszy próg': r['best_threshold'],
                'F1': r['best_f1'],
                'ROC AUC': r['roc_auc'],
                'Liczba cech': r['feature_count']
            }
            for r in model_results
        ])
        
        models_df.to_csv(f"{summary_dir}/model_results_summary.csv", index=False)
        
        plt.figure(figsize=(12, 8))
        barplot = sns.barplot(x='Model', y='F1', hue='Cechy zaawansowane', data=models_df)
        plt.title('Porównanie modeli (F1)')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)
        for i, p in enumerate(barplot.patches):
            height = p.get_height()
            barplot.text(p.get_x() + p.get_width()/2., height + 0.01, f'{height:.4f}',
                        ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{summary_dir}/model_comparison_f1.png")
        
        plt.figure(figsize=(12, 8))
        barplot = sns.barplot(x='Model', y='ROC AUC', hue='Cechy zaawansowane', data=models_df)
        plt.title('Porównanie modeli (ROC AUC)')
        plt.ylabel('ROC AUC')
        plt.xticks(rotation=45)
        for i, p in enumerate(barplot.patches):
            height = p.get_height()
            barplot.text(p.get_x() + p.get_width()/2., height + 0.01, f'{height:.4f}',
                        ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{summary_dir}/model_comparison_roc_auc.png")
    
    if adaptive_results:
        adaptive_df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'F1 (adaptacyjne)': r['adaptive_results']['f1'],
                'Precision (adaptacyjne)': r['adaptive_results']['precision'],
                'Recall (adaptacyjne)': r['adaptive_results']['recall'],
                'F1 (najlepszy stały)': r['best_fixed_results']['f1'],
                'Precision (najlepszy stały)': r['best_fixed_results']['precision'],
                'Recall (najlepszy stały)': r['best_fixed_results']['recall'],
                'Poprawa F1 (%)': (r['adaptive_results']['f1'] - r['best_fixed_results']['f1']) / r['best_fixed_results']['f1'] * 100
            }
            for r in adaptive_results
        ])
        
        adaptive_df.to_csv(f"{summary_dir}/adaptive_thresholds_summary.csv", index=False)
        
        plt.figure(figsize=(12, 8))
        
        models = adaptive_df['Model'].tolist()
        adaptive_f1 = adaptive_df['F1 (adaptacyjne)'].tolist()
        fixed_f1 = adaptive_df['F1 (najlepszy stały)'].tolist()
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width/2, fixed_f1, width, label='Stały próg')
        rects2 = ax.bar(x + width/2, adaptive_f1, width, label='Adaptacyjne progi')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('F1 Score')
        ax.set_title('Porównanie F1 Score')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        plt.savefig(f"{summary_dir}/adaptive_vs_fixed_f1.png")
        
        plt.figure(figsize=(12, 8))
        barplot = sns.barplot(x='Model', y='Poprawa F1 (%)', data=adaptive_df)
        plt.title('Poprawa F1 Score (%)')
        plt.ylabel('Poprawa F1 (%)')
        plt.xticks(rotation=45)
        for i, p in enumerate(barplot.patches):
            height = p.get_height()
            barplot.text(p.get_x() + p.get_width()/2., height + 0.5, f'{height:.2f}%',
                        ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{summary_dir}/f1_improvement.png")
    
    print(f"\nWyniki zapisane w: {summary_dir}")

def main():
    parser = argparse.ArgumentParser(description='Testy algorytmów wykrywania fraudów')
    parser.add_argument('--output_dir', type=str, default='models/full_test_results',
                      help='Katalog wyjściowy')
    parser.add_argument('--skip_training', action='store_true',
                      help='Pomiń trenowanie')
    args = parser.parse_args()
    
    if not args.skip_training:
        train_all_models(args.output_dir)
    
    model_dirs = [
        f"{args.output_dir}/randomforest_basic",
        f"{args.output_dir}/randomforest_advanced",
        f"{args.output_dir}/xgboost_basic",
        f"{args.output_dir}/xgboost_advanced",
        f"{args.output_dir}/lightgbm_basic",
        f"{args.output_dir}/lightgbm_advanced"
    ]
    
    adaptive_dirs = [f"{args.output_dir}/adaptive_thresholds/{os.path.basename(d)}" for d in model_dirs]
    
    test_adaptive_thresholds(model_dirs, f"{args.output_dir}/adaptive_thresholds")
    collect_results(model_dirs, adaptive_dirs, args.output_dir)

if __name__ == "__main__":
    main() 