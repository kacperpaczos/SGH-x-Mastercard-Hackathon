#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skrypt do łatwego uruchomienia prezentacji wyników
"""

import os
import argparse
import subprocess

def main():
    print("=" * 80)
    print("  URUCHAMIANIE PREZENTACJI WYNIKÓW DETEKCJI FRAUDÓW  ")
    print("=" * 80)
    
    parser = argparse.ArgumentParser(description='Prezentacja wyników detekcji fraudów')
    parser.add_argument('--model', type=str, default='enhanced',
                      help='Typ modelu (enhanced, simple, xgboost, lightgbm)')
    parser.add_argument('--output_dir', type=str, default='wyniki_prezentacji',
                      help='Katalog wynikowy')
    parser.add_argument('--sample_size', type=int, default=None,
                      help='Wielkość próbki')
    parser.add_argument('--quick', action='store_true',
                      help='Szybki tryb (10000 transakcji)')
    args = parser.parse_args()
    
    if args.quick and args.sample_size is None:
        args.sample_size = 10000
        print(f"Tryb szybki: {args.sample_size} transakcji")
    
    model_paths = {
        'enhanced': 'models/enhanced_model/enhanced_model.joblib',
        'simple': 'models/simple_model/basic_model.joblib',
        'xgboost': 'models/xgboost_model/xgboost_model.joblib',
        'lightgbm': 'models/lightgbm_model/lightgbm_model.joblib'
    }
    
    preprocessor_paths = {
        'enhanced': 'models/enhanced_model/enhanced_preprocessor.joblib',
        'simple': None,
        'xgboost': 'models/xgboost_model/xgboost_preprocessor.joblib',
        'lightgbm': 'models/lightgbm_model/lightgbm_preprocessor.joblib'
    }
    
    if args.model not in model_paths:
        print(f"Błąd: Nieznany typ modelu '{args.model}'")
        print(f"Dostępne typy: {', '.join(model_paths.keys())}")
        return
    
    model_path = model_paths[args.model]
    preprocessor_path = preprocessor_paths[args.model]
    
    if not os.path.exists(model_path):
        print(f"Błąd: Model '{model_path}' nie istnieje")
        return
    
    output_dir = f"{args.output_dir}_{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = ["python", "scripts/prezentacja_wynikow.py", 
           "--model_path", model_path,
           "--output_dir", output_dir]
    
    if preprocessor_path:
        cmd.extend(["--preprocessor_path", preprocessor_path])
    
    if args.sample_size:
        cmd.extend(["--sample_size", str(args.sample_size)])
    
    print(f"Model: {args.model}")
    print(f"Ścieżka: {model_path}")
    if preprocessor_path:
        print(f"Preprocessor: {preprocessor_path}")
    print(f"Wyniki: {output_dir}")
    print(f"Próbka: {args.sample_size or 'cały zbiór'}")
    
    print("\nUruchamianie prezentacji...\n")
    
    try:
        subprocess.run(cmd)
        
        print("\n" + "=" * 80)
        print("  PREZENTACJA ZAKOŃCZONA  ")
        print("=" * 80)
        print(f"Wyniki: {output_dir}")
        print("\nPolecenia:")
        print(f"  ls {output_dir}")
        print(f"  cat {output_dir}/threshold_results.json")
        print(f"  open {output_dir}/fraud_by_channel.png")
        
    except Exception as e:
        print(f"Błąd: {e}")

if __name__ == "__main__":
    main() 