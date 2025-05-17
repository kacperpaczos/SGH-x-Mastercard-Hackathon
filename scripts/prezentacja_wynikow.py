#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skrypt do prezentacji wyników modeli na pełnym zbiorze danych
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score

# Style dla wykresów
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# Wczytywanie danych
def load_data(data_path='.'):
    """Wczytuje dane z plików CSV i JSON."""
    print("Wczytywanie danych...")
    
    # Wczytanie danych o użytkownikach
    users_df = pd.read_csv(os.path.join(data_path, 'data/users.csv'))
    print(f"Załadowano dane użytkowników: {users_df.shape[0]} wierszy, {users_df.shape[1]} kolumn")
    
    # Wczytanie danych o sprzedawcach
    merchants_df = pd.read_csv(os.path.join(data_path, 'data/merchants.csv'))
    print(f"Załadowano dane sprzedawców: {merchants_df.shape[0]} wierszy, {merchants_df.shape[1]} kolumn")
    
    # Wczytanie danych o transakcjach (z pliku JSONL)
    transactions_data = []
    with open(os.path.join(data_path, 'data/transactions.json'), 'r') as f:
        for line in f:
            if line.strip():  # Ignoruj puste linie
                transactions_data.append(json.loads(line))
    
    # Konwersja na DataFrame
    transactions_df = pd.DataFrame(transactions_data)
    print(f"Załadowano dane transakcji: {transactions_df.shape[0]} wierszy, {transactions_df.shape[1]} kolumn")
    
    return users_df, merchants_df, transactions_df

def prepare_data(transactions_df, users_df, merchants_df, create_advanced_features=True):
    """Przygotowuje dane z podstawowymi i zaawansowanymi cechami."""
    print(f"Przygotowanie danych z {'zaawansowanymi' if create_advanced_features else 'podstawowymi'} cechami...")
    
    # Konwersja timestamp na datetime
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
    
    # Dodanie podstawowych cech czasowych
    transactions_df['hour'] = transactions_df['timestamp'].dt.hour
    transactions_df['day_of_week'] = transactions_df['timestamp'].dt.dayofweek
    
    # Wyodrębnienie danych o lokalizacji
    transactions_df['latitude'] = transactions_df['location'].apply(lambda x: x['lat'] if isinstance(x, dict) and 'lat' in x else None)
    transactions_df['longitude'] = transactions_df['location'].apply(lambda x: x['long'] if isinstance(x, dict) and 'long' in x else None)
    
    # Zaawansowane cechy (opcjonalnie)
    if create_advanced_features:
        # Pora dnia (morning: 5-11, day: 12-16, evening: 17-21, night: 22-4)
        transactions_df['time_of_day'] = pd.cut(
            transactions_df['hour'],
            bins=[-1, 4, 11, 16, 21, 23],
            labels=['night', 'morning', 'day', 'evening', 'night']
        )
        
        # Weekend czy dzień roboczy
        transactions_df['is_weekend'] = (transactions_df['day_of_week'] >= 5).astype(int)
        
        # Godziny szczytu (7-9, 16-18)
        transactions_df['is_rush_hour'] = ((transactions_df['hour'] >= 7) & (transactions_df['hour'] <= 9) | 
                                         (transactions_df['hour'] >= 16) & (transactions_df['hour'] <= 18)).astype(int)
    
    # Łączenie danych z użytkownikami i sprzedawcami
    merged_data = transactions_df.merge(users_df[['user_id', 'age', 'sex', 'risk_score']], 
                                      on='user_id', how='left')
    merged_data = merged_data.merge(merchants_df[['merchant_id', 'category', 'trust_score', 'has_fraud_history']], 
                                   on='merchant_id', how='left')
    
    return merged_data

def get_feature_list(include_advanced=True):
    """Zwraca listę cech w zależności od wybranych opcji."""
    # Podstawowe cechy
    features = [
        # Cechy transakcji
        'amount', 'channel', 'device', 'payment_method', 
        'is_international', 'session_length_seconds', 'is_first_time_merchant',
        'hour', 'day_of_week', 'latitude', 'longitude',
        
        # Cechy użytkownika
        'age', 'sex', 'risk_score',
        
        # Cechy sprzedawcy
        'category', 'trust_score', 'has_fraud_history'
    ]
    
    # Zaawansowane cechy (opcjonalnie)
    if include_advanced:
        advanced_features = [
            'time_of_day', 'is_weekend', 'is_rush_hour'
        ]
        features.extend(advanced_features)
    
    return features

def get_categorical_features(feature_list):
    """Zwraca listę cech kategorycznych z pełnej listy cech."""
    categorical_features = [
        'channel', 'device', 'payment_method', 'sex', 'category', 
        'time_of_day', 'is_weekend', 'is_rush_hour'
    ]
    # Filtrujemy tylko te, które są w pełnej liście cech
    return [f for f in categorical_features if f in feature_list]

def load_model(model_path, preprocessor_path=None):
    """Wczytuje model i preprocessor."""
    print(f"Wczytywanie modelu z {model_path}...")
    
    try:
        if preprocessor_path:
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            print("Wczytano model i preprocessor.")
            return model, preprocessor
        else:
            pipeline = joblib.load(model_path)
            print("Wczytano pipeline modelu.")
            return pipeline, None
    except Exception as e:
        print(f"Błąd wczytywania modelu: {e}")
        return None, None

def test_model_on_full_data(model, preprocessor, data, output_dir, threshold=0.5, adaptive_thresholds=None):
    """Testuje model na pełnym zbiorze danych."""
    print("\n=== Testowanie modelu na pełnym zbiorze danych ===")
    
    # Utworzenie katalogu na wyniki
    os.makedirs(output_dir, exist_ok=True)
    
    # Wybór cech do modelu
    features = get_feature_list(include_advanced=True)
    
    # Sprawdzenie, które cechy istnieją w danych
    features = [f for f in features if f in data.columns]
    print(f"Używanie {len(features)} cech do testowania.")
    
    # Przygotowanie danych testowych
    X_test = data[features]
    y_test = data['is_fraud']
    
    # Przetworzenie danych
    if preprocessor is not None:
        X_test_preprocessed = preprocessor.transform(X_test)
        y_pred_proba = model.predict_proba(X_test_preprocessed)[:, 1]
    else:
        # Jeśli używamy pipeline, preprocessor jest już w modelu
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Tworzenie predykcji z różnymi progami
    basic_threshold = threshold
    y_pred_basic = (y_pred_proba >= basic_threshold).astype(int)
    
    # Raport klasyfikacji z podstawowym progiem
    print(f"\n--- Wyniki z podstawowym progiem {basic_threshold} ---")
    basic_report = classification_report(y_test, y_pred_basic, output_dict=True)
    print(classification_report(y_test, y_pred_basic))
    
    # Macierz pomyłek dla podstawowego progu
    cm = confusion_matrix(y_test, y_pred_basic)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Prawdziwa klasa')
    plt.title(f'Macierz pomyłek (próg = {basic_threshold})')
    plt.savefig(os.path.join(output_dir, "confusion_matrix_basic.png"))
    plt.close()
    
    # Krzywa ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywa ROC')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()
    
    # Krzywa Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Krzywa Precision-Recall')
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.close()
    
    # Testowanie różnych progów
    test_thresholds = np.arange(0.1, 1.0, 0.1)
    thresholds_results = {}
    
    for t in test_thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = (y_pred == y_test).mean()
        
        thresholds_results[float(t)] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy)
        }
    
    # Znalezienie najlepszego progu dla F1
    best_threshold = max(thresholds_results.items(), key=lambda x: x[1]['f1'])[0]
    best_f1 = thresholds_results[best_threshold]['f1']
    
    print(f"\nNajlepszy próg dla F1: {best_threshold:.2f} (F1 = {best_f1:.4f})")
    
    # Predykcje z najlepszym progiem
    y_pred_best = (y_pred_proba >= best_threshold).astype(int)
    best_report = classification_report(y_test, y_pred_best, output_dict=True)
    print(f"\n--- Wyniki z najlepszym progiem {best_threshold} ---")
    print(classification_report(y_test, y_pred_best))
    
    # Macierz pomyłek dla najlepszego progu
    cm_best = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Prawdziwa klasa')
    plt.title(f'Macierz pomyłek (próg = {best_threshold})')
    plt.savefig(os.path.join(output_dir, "confusion_matrix_best.png"))
    plt.close()
    
    # Wykres F1 dla różnych progów
    plt.figure(figsize=(12, 8))
    plt.plot(
        list(test_thresholds),
        [thresholds_results[t]['f1'] for t in test_thresholds],
        'r-', marker='o'
    )
    plt.xlabel('Próg')
    plt.ylabel('F1 Score')
    plt.title('F1 Score dla różnych progów')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "f1_thresholds.png"))
    plt.close()
    
    # Zapisanie wyników testowania
    with open(os.path.join(output_dir, "threshold_results.json"), 'w') as f:
        json.dump({
            'basic_threshold': basic_threshold,
            'basic_metrics': basic_report,
            'best_threshold': best_threshold,
            'best_metrics': best_report,
            'thresholds_results': {str(k): v for k, v in thresholds_results.items()},
            'roc_auc': float(roc_auc)
        }, f, indent=4)
    
    # Analiza wyników według kategorii
    print("\n=== Analiza wyników według kategorii ===")
    
    # Dodajemy prawdopodobieństwa i predykcje do danych
    result_df = data.copy()
    result_df['fraud_probability'] = y_pred_proba
    result_df['predicted_fraud'] = y_pred_best
    result_df['actual_fraud'] = y_test
    
    # Analiza według kanału transakcji
    channel_results = result_df.groupby('channel').agg({
        'fraud_probability': 'mean',
        'predicted_fraud': 'mean',
        'actual_fraud': 'mean',
        'transaction_id': 'count'
    }).reset_index()
    
    channel_results.rename(columns={
        'fraud_probability': 'średnie_prawdopodobieństwo',
        'predicted_fraud': 'przewidziane_fraudy',
        'actual_fraud': 'rzeczywiste_fraudy',
        'transaction_id': 'liczba_transakcji'
    }, inplace=True)
    
    # Wizualizacja wyników według kanału
    plt.figure(figsize=(12, 8))
    channel_plot = channel_results.sort_values('rzeczywiste_fraudy', ascending=False)
    
    bar_width = 0.35
    index = np.arange(len(channel_plot))
    
    plt.bar(index, channel_plot['rzeczywiste_fraudy'], bar_width, label='Rzeczywiste fraudy')
    plt.bar(index + bar_width, channel_plot['przewidziane_fraudy'], bar_width, label='Przewidziane fraudy')
    
    plt.xlabel('Kanał transakcji')
    plt.ylabel('Wskaźnik fraudów')
    plt.title('Porównanie rzeczywistych i przewidzianych fraudów według kanału')
    plt.xticks(index + bar_width/2, channel_plot['channel'])
    plt.legend()
    plt.savefig(os.path.join(output_dir, "fraud_by_channel.png"))
    plt.close()
    
    # Zapisanie wyników do CSV
    channel_results.to_csv(os.path.join(output_dir, "fraud_by_channel.csv"), index=False)
    
    # Analiza według kategorii sprzedawcy
    if 'category' in result_df.columns:
        category_results = result_df.groupby('category').agg({
            'fraud_probability': 'mean',
            'predicted_fraud': 'mean',
            'actual_fraud': 'mean',
            'transaction_id': 'count'
        }).reset_index()
        
        category_results.rename(columns={
            'fraud_probability': 'średnie_prawdopodobieństwo',
            'predicted_fraud': 'przewidziane_fraudy',
            'actual_fraud': 'rzeczywiste_fraudy',
            'transaction_id': 'liczba_transakcji'
        }, inplace=True)
        
        # Zapisanie wyników do CSV
        category_results.to_csv(os.path.join(output_dir, "fraud_by_category.csv"), index=False)
        
        # Wizualizacja dla top 10 kategorii z najwyższym wskaźnikiem fraudów
        top_categories = category_results.sort_values('rzeczywiste_fraudy', ascending=False).head(10)
        
        plt.figure(figsize=(14, 8))
        bar_width = 0.35
        index = np.arange(len(top_categories))
        
        plt.bar(index, top_categories['rzeczywiste_fraudy'], bar_width, label='Rzeczywiste fraudy')
        plt.bar(index + bar_width, top_categories['przewidziane_fraudy'], bar_width, label='Przewidziane fraudy')
        
        plt.xlabel('Kategoria sprzedawcy')
        plt.ylabel('Wskaźnik fraudów')
        plt.title('Porównanie fraudów według top 10 kategorii sprzedawców')
        plt.xticks(index + bar_width/2, top_categories['category'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fraud_by_category_top10.png"))
        plt.close()
    
    # Analiza według pory dnia i dnia tygodnia
    if 'hour' in result_df.columns and 'day_of_week' in result_df.columns:
        # Przygotowanie danych
        hour_results = result_df.groupby('hour').agg({
            'fraud_probability': 'mean',
            'predicted_fraud': 'mean',
            'actual_fraud': 'mean',
            'transaction_id': 'count'
        }).reset_index()
        
        day_results = result_df.groupby('day_of_week').agg({
            'fraud_probability': 'mean',
            'predicted_fraud': 'mean',
            'actual_fraud': 'mean',
            'transaction_id': 'count'
        }).reset_index()
        
        # Mapa nazw dni tygodnia
        day_names = {
            0: 'Poniedziałek',
            1: 'Wtorek',
            2: 'Środa',
            3: 'Czwartek',
            4: 'Piątek',
            5: 'Sobota',
            6: 'Niedziela'
        }
        day_results['day_name'] = day_results['day_of_week'].map(day_names)
        
        # Wizualizacja według godziny
        plt.figure(figsize=(14, 8))
        plt.plot(hour_results['hour'], hour_results['actual_fraud'], 'b-', marker='o', label='Rzeczywiste fraudy')
        plt.plot(hour_results['hour'], hour_results['predicted_fraud'], 'r-', marker='x', label='Przewidziane fraudy')
        plt.xlabel('Godzina dnia')
        plt.ylabel('Wskaźnik fraudów')
        plt.title('Wskaźnik fraudów według godziny dnia')
        plt.xticks(range(24))
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "fraud_by_hour.png"))
        plt.close()
        
        # Wizualizacja według dnia tygodnia
        plt.figure(figsize=(14, 8))
        plt.plot(day_results['day_of_week'], day_results['actual_fraud'], 'b-', marker='o', label='Rzeczywiste fraudy')
        plt.plot(day_results['day_of_week'], day_results['predicted_fraud'], 'r-', marker='x', label='Przewidziane fraudy')
        plt.xlabel('Dzień tygodnia')
        plt.ylabel('Wskaźnik fraudów')
        plt.title('Wskaźnik fraudów według dnia tygodnia')
        plt.xticks(day_results['day_of_week'], day_results['day_name'], rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fraud_by_day.png"))
        plt.close()
    
    print(f"\nWyniki analizy zostały zapisane w katalogu: {output_dir}")
    return {
        'basic_threshold': basic_threshold,
        'basic_f1': basic_report['1']['f1-score'],
        'best_threshold': best_threshold,
        'best_f1': best_report['1']['f1-score'],
        'roc_auc': roc_auc
    }

def main():
    parser = argparse.ArgumentParser(description='Prezentacja wyników modelu na pełnym zbiorze danych.')
    parser.add_argument('--model_path', type=str, default='models/enhanced_model/enhanced_model.joblib',
                      help='Ścieżka do modelu (domyślnie: models/enhanced_model/enhanced_model.joblib)')
    parser.add_argument('--preprocessor_path', type=str, default=None,
                      help='Ścieżka do preprocessora (opcjonalnie)')
    parser.add_argument('--data_path', type=str, default='.',
                      help='Ścieżka do katalogu z danymi (domyślnie: .)')
    parser.add_argument('--output_dir', type=str, default='wyniki_prezentacji',
                      help='Katalog wyjściowy (domyślnie: wyniki_prezentacji)')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Podstawowy próg decyzyjny (domyślnie: 0.5)')
    parser.add_argument('--sample_size', type=int, default=None,
                      help='Liczba przykładów do analizy (domyślnie: wszystkie)')
    parser.add_argument('--no_advanced_features', action='store_true',
                      help='Nie używaj zaawansowanych cech czasowych')
    args = parser.parse_args()
    
    print("=" * 80)
    print("  PREZENTACJA WYNIKÓW MODELU WYKRYWANIA FRAUDÓW  ")
    print("=" * 80)
    
    # Wczytanie danych
    users_df, merchants_df, transactions_df = load_data(args.data_path)
    
    # Przygotowanie danych
    use_advanced = not args.no_advanced_features
    data = prepare_data(transactions_df, users_df, merchants_df, create_advanced_features=use_advanced)
    
    # Dla dużych zbiorów danych, losujemy próbkę
    if args.sample_size is not None and args.sample_size < len(data):
        print(f"Losowanie próbki {args.sample_size} przykładów...")
        sample_data = data.sample(args.sample_size, random_state=42)
    else:
        sample_data = data
        print(f"Używanie pełnego zbioru danych: {len(data)} przykładów.")
    
    # Wczytanie modelu
    model, preprocessor = load_model(args.model_path, args.preprocessor_path)
    if model is None:
        print("Nie udało się wczytać modelu. Koniec pracy.")
        return
    
    # Testowanie modelu na danych
    results = test_model_on_full_data(
        model, 
        preprocessor, 
        sample_data, 
        args.output_dir, 
        threshold=args.threshold
    )
    
    # Podsumowanie
    print("\n" + "=" * 80)
    print("  PODSUMOWANIE WYNIKÓW  ")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Rozmiar danych: {len(sample_data)} transakcji")
    print(f"Podstawowy próg: {results['basic_threshold']}, F1: {results['basic_f1']:.4f}")
    print(f"Najlepszy próg: {results['best_threshold']}, F1: {results['best_f1']:.4f}")
    print(f"Poprawa F1: {100 * (results['best_f1'] - results['basic_f1']) / results['basic_f1']:.2f}%")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    
    print(f"\nWszystkie wyniki zapisane w: {args.output_dir}")
    print("\nAby uzyskać szczegółowe wyniki, sprawdź pliki CSV i obrazy w katalogu wynikowym.")

if __name__ == "__main__":
    main() 