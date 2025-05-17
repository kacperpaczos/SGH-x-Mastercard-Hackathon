#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skrypt do testowania modelu z adaptacyjnymi progami dla różnych kombinacji cech.
Umożliwia porównanie standardowego podejścia z jednym progiem z podejściem używającym
różnych progów dla różnych kombinacji cech.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse
import random
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

# Import funkcji z enhanced_model.py
from enhanced_model import (
    load_data, parse_timestamp, extract_location_features, 
    create_user_aggregates, create_merchant_aggregates,
    prepare_data, get_feature_list, get_categorical_features
)

def load_model_and_preprocessor(model_path, preprocessor_path=None):
    """Wczytuje model i preprocessor."""
    print(f"Wczytywanie modelu z {model_path}...")
    
    # Sprawdź czy model jest pełnym pipelineiem czy samym modelem
    model = joblib.load(model_path)
    
    if hasattr(model, 'transform') and hasattr(model, 'predict'):
        # Model to pełny pipeline
        return model, None
    
    # W przeciwnym razie potrzebujemy osobno preprocessora
    if preprocessor_path is None:
        raise ValueError("Preprocessor path must be provided when model does not include preprocessing.")
    
    print(f"Wczytywanie preprocessora z {preprocessor_path}...")
    preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor

def find_optimal_thresholds_for_combinations(X_test, y_test, y_pred_proba, feature_combinations, output_dir='models'):
    """
    Znajduje optymalne progi decyzyjne dla różnych kombinacji cech.
    
    Parametry:
    ----------
    X_test : DataFrame
        Dane testowe
    y_test : Series
        Etykiety testowe
    y_pred_proba : array
        Przewidywane prawdopodobieństwa
    feature_combinations : list
        Lista list cech, dla których będziemy szukać optymalnych progów
    output_dir : str
        Katalog wyjściowy do zapisania wyników
    
    Zwraca:
    -------
    thresholds_dict : dict
        Słownik zawierający optymalne progi dla różnych kombinacji cech
    """
    print("\nZnajdowanie optymalnych progów dla kombinacji cech...")
    
    thresholds_dict = {}
    
    for feature_list in feature_combinations:
        # Tworzymy nową kolumnę będącą kombinacją wartości cech
        X_test['combo'] = X_test[feature_list].apply(lambda row: '_'.join([str(row[f]) for f in feature_list]), axis=1)
        
        # Dla każdej unikalnej kombinacji wartości cech
        combinations = X_test['combo'].unique()
        
        combo_results = {}
        for combo in combinations:
            # Wybieramy indeksy przykładów o danej kombinacji cech
            combo_indices = X_test['combo'] == combo
            
            if sum(combo_indices) < 50:  # Pomijamy kombinacje z małą liczbą przykładów
                continue
            
            # Filtrujemy dane testowe do wybranej kombinacji
            y_test_filtered = y_test[combo_indices]
            y_proba_filtered = y_pred_proba[combo_indices]
            
            # Jeśli wszystkie przykłady mają tę samą etykietę, nie możemy znaleźć optymalnego progu
            if y_test_filtered.nunique() < 2 or sum(y_test_filtered) < 5:
                continue
            
            # Obliczamy różne miary dla różnych progów
            thresholds = np.linspace(0.1, 0.9, 17)  # Z krokiem 0.05
            f1_scores = []
            
            for threshold in thresholds:
                y_pred = (y_proba_filtered >= threshold).astype(int)
                # Sprawdzamy, czy mamy wystarczająco dużo pozytywnych predykcji dla obliczenia f1
                if sum(y_pred) > 0 and sum(y_test_filtered) > 0:
                    f1 = f1_score(y_test_filtered, y_pred)
                    f1_scores.append(f1)
                else:
                    f1_scores.append(0)
            
            # Znajdź optymalny próg
            if max(f1_scores) > 0:
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx]
                best_f1 = f1_scores[best_idx]
                
                # Oblicz precision i recall dla najlepszego progu
                y_pred_best = (y_proba_filtered >= best_threshold).astype(int)
                precision = precision_score(y_test_filtered, y_pred_best, zero_division=0)
                recall = recall_score(y_test_filtered, y_pred_best, zero_division=0)
                
                # Zapisz wynik
                combo_results[combo] = {
                    'optimal_threshold': float(best_threshold),
                    'f1_score': float(best_f1),
                    'precision': float(precision),
                    'recall': float(recall),
                    'sample_count': int(sum(combo_indices)),
                    'fraud_rate': float(y_test_filtered.mean())
                }
        
        # Zapisz wyniki dla tej kombinacji cech
        combo_name = "_".join(feature_list)
        thresholds_dict[combo_name] = combo_results
        
        # Zapisanie wyników do pliku
        with open(os.path.join(output_dir, f"optimal_thresholds_{combo_name}.json"), 'w') as f:
            json.dump(combo_results, f, indent=4)
        
        # Wizualizacja jak wskaźnik fraudów wpływa na optymalny próg
        plt.figure(figsize=(10, 6))
        fraud_rates = [combo_results[c]['fraud_rate'] for c in combo_results]
        opt_thresholds = [combo_results[c]['optimal_threshold'] for c in combo_results]
        
        plt.scatter(fraud_rates, opt_thresholds, alpha=0.7)
        plt.xlabel('Wskaźnik fraudów w danej kombinacji')
        plt.ylabel('Optymalny próg')
        plt.title(f'Zależność między wskaźnikiem fraudów a optymalnym progiem\nKombinacja cech: {combo_name}')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"fraud_rate_vs_threshold_{combo_name}.png"))
        
        # Wizualizacja rozkładu optymalnych progów
        plt.figure(figsize=(10, 6))
        plt.hist(opt_thresholds, bins=20, alpha=0.7)
        plt.xlabel('Optymalny próg')
        plt.ylabel('Liczba kombinacji')
        plt.title(f'Rozkład optymalnych progów\nKombinacja cech: {combo_name}')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"threshold_distribution_{combo_name}.png"))
    
    return thresholds_dict

def apply_adaptive_thresholds(X_test, y_test, y_pred_proba, thresholds_dict, default_threshold=0.5):
    """
    Stosuje adaptacyjne progi decyzyjne na podstawie kombinacji cech.
    
    Parametry:
    ----------
    X_test : DataFrame
        Dane testowe
    y_test : Series
        Etykiety testowe
    y_pred_proba : array
        Przewidywane prawdopodobieństwa
    thresholds_dict : dict
        Słownik zawierający optymalne progi dla różnych kombinacji cech
    default_threshold : float
        Domyślny próg do zastosowania, gdy nie ma optymalnego progu dla danej kombinacji
    
    Zwraca:
    -------
    y_pred : array
        Predykcje z zastosowaniem adaptacyjnych progów
    results : dict
        Wyniki ewaluacji
    """
    print("\nStosowanie adaptacyjnych progów...")
    
    # Inicjalizacja tablicy predykcji
    y_pred = np.zeros(len(y_test), dtype=int)
    
    # Słownik do śledzenia użytych progów
    used_thresholds = {}
    
    # Dla każdego przykładu
    for i in range(len(X_test)):
        applied_threshold = default_threshold
        applied_combo = "default"
        
        # Dla każdej kombinacji cech
        for combo_name, combo_thresholds in thresholds_dict.items():
            # Pobierz cechy w tej kombinacji
            features = combo_name.split('_')
            
            # Utwórz wartość kombinacji dla bieżącego przykładu
            combo_value = '_'.join([str(X_test.iloc[i][f]) for f in features])
            
            # Sprawdź czy mamy optymalny próg dla tej kombinacji
            if combo_value in combo_thresholds:
                applied_threshold = combo_thresholds[combo_value]['optimal_threshold']
                applied_combo = combo_name
                break
        
        # Zastosuj wybrany próg
        y_pred[i] = 1 if y_pred_proba[i] >= applied_threshold else 0
        
        # Śledź użyte progi
        if applied_combo not in used_thresholds:
            used_thresholds[applied_combo] = 0
        used_thresholds[applied_combo] += 1
    
    # Obliczenie metryk
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Przygotowanie wyników
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'thresholds_usage': used_thresholds
    }
    
    # Wyświetlenie wyników
    print("\n--- Wyniki z adaptacyjnymi progami ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Wyświetlenie użycia progów
    total = sum(used_thresholds.values())
    print("\nUżycie progów:")
    for combo, count in used_thresholds.items():
        print(f"  {combo}: {count} przykładów ({100*count/total:.2f}%)")
    
    return y_pred, results

def compare_with_fixed_threshold(y_test, y_pred_proba, thresholds):
    """
    Porównuje wyniki z różnymi stałymi progami.
    
    Parametry:
    ----------
    y_test : Series
        Etykiety testowe
    y_pred_proba : array
        Przewidywane prawdopodobieństwa
    thresholds : list
        Lista progów do przetestowania
    
    Zwraca:
    -------
    results : dict
        Wyniki dla różnych progów
    """
    print("\nPorównanie wyników ze stałymi progami:")
    
    results = {}
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Próg {threshold:.2f}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
        
        results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Testowanie modelu z adaptacyjnymi progami dla różnych kombinacji cech.')
    parser.add_argument('--model_path', type=str, default='models/randomforest_advanced_model.joblib',
                        help='Ścieżka do modelu (domyślnie: models/randomforest_advanced_model.joblib)')
    parser.add_argument('--preprocessor_path', type=str, default=None,
                        help='Ścieżka do preprocessora (opcjonalnie, jeśli model nie zawiera preprocessora)')
    parser.add_argument('--data_path', type=str, default='.',
                        help='Ścieżka do katalogu z danymi (domyślnie: .)')
    parser.add_argument('--output_dir', type=str, default='models/adaptive_thresholds',
                        help='Katalog wyjściowy do zapisania wyników (domyślnie: models/adaptive_thresholds)')
    parser.add_argument('--default_threshold', type=float, default=0.2,
                        help='Domyślny próg decyzyjny (domyślnie: 0.2)')
    parser.add_argument('--test_size', type=int, default=50000,
                        help='Liczba przykładów do testowania (domyślnie: 50000)')
    args = parser.parse_args()
    
    # Utworzenie katalogu wyjściowego
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Wczytanie modelu i preprocessora
    pipeline, preprocessor = load_model_and_preprocessor(args.model_path, args.preprocessor_path)
    
    # Wczytanie informacji o modelu
    model_info_path = args.model_path.replace('_model.joblib', '_results.json')
    use_advanced_features = True
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        use_advanced_features = model_info.get('use_advanced_features', True)
        print(f"Używanie {'zaawansowanych' if use_advanced_features else 'podstawowych'} cech na podstawie konfiguracji modelu.")
    
    # Wczytanie danych
    users_df, merchants_df, transactions_df = load_data(args.data_path)
    
    # Przygotowanie danych z odpowiednimi cechami
    print(f"Przygotowanie danych z {'zaawansowanymi' if use_advanced_features else 'podstawowymi'} cechami...")
    data = prepare_data(transactions_df, users_df, merchants_df, create_advanced_features=use_advanced_features)
    
    # Losowe wybranie przykładów do testowania
    if args.test_size < len(data):
        test_data = data.sample(args.test_size, random_state=42)
    else:
        test_data = data
    
    print(f"Używanie {len(test_data)} przykładów do testowania.")
    
    # Pobieranie cech odpowiednich dla modelu
    features = get_feature_list(include_advanced=use_advanced_features)
    categorical_features = get_categorical_features(features)
    print(f"Używam {len(features)} cech do testowania (zgodnie z modelem).")
    
    # Przygotowanie danych do testowania
    X_test = test_data[features]
    y_test = test_data['is_fraud']
    
    # Predykcje prawdopodobieństw
    print("Wykonywanie predykcji...")
    if preprocessor is not None:
        X_test_preprocessed = preprocessor.transform(X_test)
        y_pred_proba = pipeline.predict_proba(X_test_preprocessed)[:, 1]
    else:
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Porównanie wyników ze stałymi progami
    fixed_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fixed_results = compare_with_fixed_threshold(y_test, y_pred_proba, fixed_thresholds)
    
    # Zapisanie wyników ze stałymi progami
    with open(os.path.join(args.output_dir, "fixed_threshold_results.json"), 'w') as f:
        json.dump(fixed_results, f, indent=4)
    
    # Definicja kombinacji cech do testowania
    feature_combinations = [
        ['channel', 'payment_method'],
        ['category', 'channel'],
        ['day_of_week', 'hour'],
        ['channel', 'is_weekend'],
        ['payment_method', 'is_international'],
        ['channel', 'payment_method', 'category']
    ]
    
    # Znalezienie optymalnych progów dla kombinacji cech
    thresholds_dict = find_optimal_thresholds_for_combinations(
        X_test.copy(), y_test, y_pred_proba, feature_combinations, output_dir=args.output_dir
    )
    
    # Zastosowanie adaptacyjnych progów
    adaptive_y_pred, adaptive_results = apply_adaptive_thresholds(
        X_test, y_test, y_pred_proba, thresholds_dict, default_threshold=args.default_threshold
    )
    
    # Zapisanie wyników adaptacyjnych progów
    with open(os.path.join(args.output_dir, "adaptive_thresholds_results.json"), 'w') as f:
        json.dump({
            'adaptive_results': adaptive_results,
            'best_fixed_threshold': max(fixed_results.items(), key=lambda x: x[1]['f1'])[0],
            'best_fixed_results': max(fixed_results.items(), key=lambda x: x[1]['f1'])[1]
        }, f, indent=4)
    
    # Porównanie z najlepszym stałym progiem
    best_threshold = max(fixed_results.items(), key=lambda x: x[1]['f1'])[0]
    best_fixed_f1 = fixed_results[best_threshold]['f1']
    
    print("\n--- Porównanie najlepszych wyników ---")
    print(f"Najlepszy stały próg: {best_threshold} - F1: {best_fixed_f1:.4f}")
    print(f"Adaptacyjne progi - F1: {adaptive_results['f1']:.4f}")
    
    improvement = (adaptive_results['f1'] - best_fixed_f1) / best_fixed_f1 * 100
    print(f"Poprawa F1: {improvement:.2f}%")
    
    # Wizualizacja porównania
    plt.figure(figsize=(10, 6))
    
    # Precision
    precisions = [fixed_results[t]['precision'] for t in fixed_thresholds]
    plt.plot(fixed_thresholds, precisions, 'b-', label='Precision (stały próg)')
    plt.axhline(y=adaptive_results['precision'], color='b', linestyle='--', label='Precision (adaptacyjne progi)')
    
    # Recall
    recalls = [fixed_results[t]['recall'] for t in fixed_thresholds]
    plt.plot(fixed_thresholds, recalls, 'g-', label='Recall (stały próg)')
    plt.axhline(y=adaptive_results['recall'], color='g', linestyle='--', label='Recall (adaptacyjne progi)')
    
    # F1
    f1s = [fixed_results[t]['f1'] for t in fixed_thresholds]
    plt.plot(fixed_thresholds, f1s, 'r-', label='F1 (stały próg)')
    plt.axhline(y=adaptive_results['f1'], color='r', linestyle='--', label='F1 (adaptacyjne progi)')
    
    plt.xlabel('Próg')
    plt.ylabel('Wartość')
    plt.title('Porównanie stałych i adaptacyjnych progów')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "adaptive_vs_fixed_thresholds.png"))
    
    print("\nTestowanie zakończone! Wyniki zapisane w katalogu:", args.output_dir)

if __name__ == "__main__":
    main() 