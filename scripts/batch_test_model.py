#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skrypt do testowania modelu na wielu transakcjach
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
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score

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

def parse_timestamp(df):
    """Konwertuje kolumnę timestamp na format datetime."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    return df

def prepare_data(transactions_df, users_df, merchants_df):
    """Przygotowuje dane do testowania modelu."""
    
    # Konwersja timestamp na datetime
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
    
    # Dodanie cech czasowych
    transactions_df['hour'] = transactions_df['timestamp'].dt.hour
    transactions_df['day_of_week'] = transactions_df['timestamp'].dt.dayofweek
    
    # Wyodrębnienie danych o lokalizacji
    transactions_df['latitude'] = transactions_df['location'].apply(lambda x: x['lat'] if isinstance(x, dict) and 'lat' in x else None)
    transactions_df['longitude'] = transactions_df['location'].apply(lambda x: x['long'] if isinstance(x, dict) and 'long' in x else None)
    
    # Łączenie danych z użytkownikami i sprzedawcami
    merged_data = transactions_df.merge(users_df[['user_id', 'age', 'sex', 'risk_score']], 
                                      on='user_id', how='left')
    merged_data = merged_data.merge(merchants_df[['merchant_id', 'category', 'trust_score', 'has_fraud_history']], 
                                   on='merchant_id', how='left')
    
    return merged_data

def generate_feature_combinations_report(test_data, model, preprocessor, output_dir='models', threshold=0.2):
    """
    Generuje szczegółowy raport o prawdopodobieństwach fraudów dla różnych kombinacji cech.
    
    Parametry:
    ----------
    test_data : DataFrame
        Dane testowe
    model : obiekt
        Wytrenowany model
    preprocessor : obiekt
        Preprocessor do transformacji cech
    output_dir : str
        Katalog wyjściowy na raporty
    threshold : float
        Próg decyzyjny dla klasyfikacji
    """
    
    # Wybór cech do modelu
    features = [
        'amount', 'channel', 'device', 'latitude', 'longitude', 'payment_method', 
        'is_international', 'session_length_seconds', 'is_first_time_merchant',
        'hour', 'day_of_week', 'age', 'sex', 'risk_score',
        'category', 'trust_score', 'has_fraud_history'
    ]
    
    # Przygotowanie danych testowych
    X_test = test_data[features]
    y_test = test_data['is_fraud']
    
    # Przetworzenie danych
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Predykcje prawdopodobieństw
    y_prob = model.predict_proba(X_test_preprocessed)[:, 1]
    
    # Zastosowanie progu decyzyjnego
    y_pred = (y_prob >= threshold).astype(int)
    
    # Dodanie wyników do danych
    result_df = test_data.copy()
    result_df['fraud_probability'] = y_prob
    result_df['predicted_fraud'] = y_pred
    result_df['actual_fraud'] = y_test
    
    # Utworzenie katalogu na wykresy, jeśli nie istnieje
    os.makedirs(os.path.join(output_dir, 'feature_reports'), exist_ok=True)
    
    print("\n--- Generowanie raportów dla różnych kombinacji cech ---")
    
    # 1. Analiza według kanału i metody płatności
    ch_pay_df = result_df.groupby(['channel', 'payment_method']).agg({
        'fraud_probability': 'mean',
        'predicted_fraud': 'mean',
        'actual_fraud': 'mean',
        'transaction_id': 'count'
    }).reset_index()
    
    ch_pay_df.rename(columns={
        'fraud_probability': 'avg_fraud_probability',
        'predicted_fraud': 'predicted_fraud_rate',
        'actual_fraud': 'actual_fraud_rate',
        'transaction_id': 'transaction_count'
    }, inplace=True)
    
    # Sortowanie według prawdopodobieństwa fraudu
    ch_pay_df.sort_values('avg_fraud_probability', ascending=False, inplace=True)
    
    # Zapisanie do CSV
    ch_pay_df.to_csv(os.path.join(output_dir, 'feature_reports', 'channel_payment_method_report.csv'), index=False)
    
    # Wizualizacja
    plt.figure(figsize=(14, 8))
    top_combinations = ch_pay_df.head(10)
    sns.barplot(x='channel', y='avg_fraud_probability', hue='payment_method', data=top_combinations)
    plt.title('Top 10 kombinacji kanał-metoda płatności według prawdopodobieństwa fraudu')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_reports', 'top_channel_payment_combinations.png'))
    
    # 2. Analiza według kategorii sprzedawcy i kanału
    cat_ch_df = result_df.groupby(['category', 'channel']).agg({
        'fraud_probability': 'mean',
        'predicted_fraud': 'mean',
        'actual_fraud': 'mean',
        'transaction_id': 'count'
    }).reset_index()
    
    cat_ch_df.rename(columns={
        'fraud_probability': 'avg_fraud_probability',
        'predicted_fraud': 'predicted_fraud_rate',
        'actual_fraud': 'actual_fraud_rate',
        'transaction_id': 'transaction_count'
    }, inplace=True)
    
    # Sortowanie według prawdopodobieństwa fraudu
    cat_ch_df.sort_values('avg_fraud_probability', ascending=False, inplace=True)
    
    # Zapisanie do CSV
    cat_ch_df.to_csv(os.path.join(output_dir, 'feature_reports', 'category_channel_report.csv'), index=False)
    
    # Wizualizacja
    plt.figure(figsize=(14, 8))
    top_cat_combinations = cat_ch_df.head(10)
    sns.barplot(x='category', y='avg_fraud_probability', hue='channel', data=top_cat_combinations)
    plt.title('Top 10 kombinacji kategoria-kanał według prawdopodobieństwa fraudu')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_reports', 'top_category_channel_combinations.png'))
    
    # 3. Analiza według dnia tygodnia i godziny
    day_hour_df = result_df.groupby(['day_of_week', 'hour']).agg({
        'fraud_probability': 'mean',
        'predicted_fraud': 'mean',
        'actual_fraud': 'mean',
        'transaction_id': 'count'
    }).reset_index()
    
    day_hour_df.rename(columns={
        'fraud_probability': 'avg_fraud_probability',
        'predicted_fraud': 'predicted_fraud_rate',
        'actual_fraud': 'actual_fraud_rate',
        'transaction_id': 'transaction_count'
    }, inplace=True)
    
    # Sortowanie według prawdopodobieństwa fraudu
    day_hour_df.sort_values('avg_fraud_probability', ascending=False, inplace=True)
    
    # Zapisanie do CSV
    day_hour_df.to_csv(os.path.join(output_dir, 'feature_reports', 'day_hour_report.csv'), index=False)
    
    # Wizualizacja jako heatmapa
    day_hour_pivot = day_hour_df.pivot(index='day_of_week', columns='hour', values='avg_fraud_probability')
    
    # Uporządkowanie dni tygodnia
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_hour_pivot = day_hour_pivot.reindex(day_order)
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(day_hour_pivot, cmap='YlOrRd', annot=False, fmt='.3f')
    plt.title('Prawdopodobieństwo fraudu według dnia tygodnia i godziny')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_reports', 'day_hour_heatmap.png'))
    
    # 4. Analiza według kwoty transakcji
    # Tworzenie binów dla kwot
    bins = [0, 25, 50, 100, 200, 500, result_df['amount'].max() + 1]
    labels = ['0-25', '25-50', '50-100', '100-200', '200-500', '500+']
    
    result_df['amount_bin'] = pd.cut(result_df['amount'], bins=bins, labels=labels)
    
    amount_df = result_df.groupby('amount_bin').agg({
        'fraud_probability': 'mean',
        'predicted_fraud': 'mean',
        'actual_fraud': 'mean',
        'transaction_id': 'count'
    }).reset_index()
    
    amount_df.rename(columns={
        'fraud_probability': 'avg_fraud_probability',
        'predicted_fraud': 'predicted_fraud_rate',
        'actual_fraud': 'actual_fraud_rate',
        'transaction_id': 'transaction_count'
    }, inplace=True)
    
    # Zapisanie do CSV
    amount_df.to_csv(os.path.join(output_dir, 'feature_reports', 'amount_report.csv'), index=False)
    
    # Wizualizacja
    plt.figure(figsize=(10, 6))
    sns.barplot(x='amount_bin', y='avg_fraud_probability', data=amount_df, color='darkred')
    plt.title('Prawdopodobieństwo fraudu według kwoty transakcji')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_reports', 'amount_fraud_probability.png'))
    
    # 5. Podsumowanie top 20 najbardziej podejrzanych kombinacji cech
    # Łączymy kanał, metodę płatności i kategorię
    result_df['channel_payment_category'] = result_df['channel'] + '_' + result_df['payment_method'] + '_' + result_df['category']
    
    channel_payment_category_df = result_df.groupby('channel_payment_category').agg({
        'fraud_probability': 'mean',
        'predicted_fraud': 'mean',
        'actual_fraud': 'mean',
        'transaction_id': 'count'
    }).reset_index()
    
    channel_payment_category_df.rename(columns={
        'fraud_probability': 'avg_fraud_probability',
        'predicted_fraud': 'predicted_fraud_rate',
        'actual_fraud': 'actual_fraud_rate',
        'transaction_id': 'transaction_count'
    }, inplace=True)
    
    # Sortowanie według prawdopodobieństwa fraudu
    channel_payment_category_df.sort_values('avg_fraud_probability', ascending=False, inplace=True)
    
    # Zapisanie do CSV (tylko top 20)
    top20_combinations = channel_payment_category_df.head(20)
    top20_combinations.to_csv(os.path.join(output_dir, 'feature_reports', 'top20_suspicious_combinations.csv'), index=False)
    
    # Wizualizacja
    plt.figure(figsize=(12, 10))
    top10_combinations = channel_payment_category_df.head(10)
    sns.barplot(y='channel_payment_category', x='avg_fraud_probability', data=top10_combinations, palette='viridis')
    plt.title('Top 10 najbardziej podejrzanych kombinacji cech')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_reports', 'top10_suspicious_combinations.png'))
    
    print(f"Szczegółowe raporty zostały zapisane w katalogu {os.path.join(output_dir, 'feature_reports')}")
    return {
        'top_channel_payment': ch_pay_df.head(5).to_dict('records'),
        'top_category_channel': cat_ch_df.head(5).to_dict('records'),
        'top_day_hour': day_hour_df.head(5).to_dict('records'),
        'amount_fraud': amount_df.to_dict('records')
    }

def test_model_on_batch(test_data, model, preprocessor, threshold=0.5, sample_size=None, random_seed=42):
    """
    Testuje model na zestawie wielu transakcji.
    
    Parametry:
    ----------
    test_data : DataFrame
        Dane testowe
    model : obiekt
        Wytrenowany model
    preprocessor : obiekt
        Preprocessor do transformacji cech
    threshold : float
        Próg decyzyjny dla klasyfikacji
    sample_size : int lub None
        Liczba losowych transakcji do testowania (None = wszystkie)
    random_seed : int
        Ziarno losowości
    
    Zwraca:
    -------
    results : dict
        Wyniki testów
    """
    
    # Wybieranie losowej próbki, jeśli określono wielkość
    if sample_size is not None and sample_size < len(test_data):
        random.seed(random_seed)
        test_data = test_data.sample(sample_size, random_state=random_seed)
    
    # Wybór cech do modelu
    features = [
        # Cechy transakcji
        'amount', 'channel', 'device', 'latitude', 'longitude', 'payment_method', 
        'is_international', 'session_length_seconds', 'is_first_time_merchant',
        'hour', 'day_of_week', 
        
        # Cechy użytkownika
        'age', 'sex', 'risk_score',
        
        # Cechy sprzedawcy
        'category', 'trust_score', 'has_fraud_history'
    ]
    
    # Przygotowanie danych testowych
    X_test = test_data[features]
    y_test = test_data['is_fraud']
    
    # Przetworzenie danych
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Predykcje prawdopodobieństw
    y_prob = model.predict_proba(X_test_preprocessed)[:, 1]
    
    # Zastosowanie progu decyzyjnego
    y_pred = (y_prob >= threshold).astype(int)
    
    # Raport klasyfikacji
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Tworzenie macierzy pomyłek
    cm = confusion_matrix(y_test, y_pred)
    
    # Wyliczenie ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Wyświetlenie raportu
    print("\n--- Raport klasyfikacji (próg = {:.2f}) ---".format(threshold))
    print(classification_report(y_test, y_pred))
    
    # Macierz pomyłek
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Prawdziwa klasa')
    plt.title(f'Macierz pomyłek (próg = {threshold:.2f})')
    plt.savefig('batch_confusion_matrix.png')
    print("Macierz pomyłek zapisana jako 'batch_confusion_matrix.png'")
    
    # Krzywa ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywa ROC')
    plt.legend(loc="lower right")
    plt.savefig('batch_roc_curve.png')
    print("Krzywa ROC zapisana jako 'batch_roc_curve.png'")
    
    # Wykres przewidywań dla różnych typów transakcji
    plt.figure(figsize=(12, 6))
    merged_result_df = test_data.copy()
    merged_result_df['fraud_probability'] = y_prob
    merged_result_df['predicted_fraud'] = y_pred
    
    # Analiza średniego prawdopodobieństwa fraudu według kanału
    plt.subplot(1, 2, 1)
    channels_prob = merged_result_df.groupby('channel')['fraud_probability'].mean().sort_values(ascending=False)
    channels_prob.plot(kind='bar', color='teal')
    plt.title('Średnie prawdopodobieństwo fraudu według kanału')
    plt.tight_layout()
    
    # Analiza średniego prawdopodobieństwa fraudu według metody płatności
    plt.subplot(1, 2, 2)
    payment_prob = merged_result_df.groupby('payment_method')['fraud_probability'].mean().sort_values(ascending=False)
    payment_prob.plot(kind='bar', color='crimson')
    plt.title('Średnie prawdopodobieństwo fraudu według metody płatności')
    plt.tight_layout()
    
    plt.savefig('batch_fraud_probability_by_features.png')
    print("Wykres prawdopodobieństwa fraudu zapisany jako 'batch_fraud_probability_by_features.png'")
    
    # Wyniki
    results = {
        'accuracy': report['accuracy'],
        'precision_fraud': report['1']['precision'],
        'recall_fraud': report['1']['recall'],
        'f1_score_fraud': report['1']['f1-score'],
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'sample_size': len(test_data),
        'threshold': threshold
    }
    
    return results

def get_features_and_target(data):
    """Zwraca cechy i etykiety."""
    # Wybór cech, które istnieją w danych
    all_features = [
        'amount', 'channel', 'device', 'payment_method', 
        'is_international', 'session_length_seconds', 
        'hour', 'day_of_week', 'latitude', 'longitude',
        'age', 'sex', 'risk_score',
        'category', 'trust_score', 'has_fraud_history'
    ]
    
    # Sprawdzenie, które cechy rzeczywiście istnieją w danych
    features = [f for f in all_features if f in data.columns]
    print(f"Używane cechy ({len(features)}): {', '.join(features)}")
    
    X = data[features]
    y = data['is_fraud']
    
    return X, y

def load_model(model_path):
    """Wczytuje model i preprocessor."""
    print(f"Wczytywanie modelu z {model_path}...")
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Błąd wczytywania modelu: {e}")
        return None

def test_thresholds(model, X_test, y_test, thresholds=None, output_dir='.', model_name='model'):
    """Testuje model z różnymi progami decyzyjnymi i zapisuje wyniki."""
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    # Predykcje prawdopodobieństw
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"Błąd podczas predykcji: {e}")
        return None
    
    # Obliczanie miar dla różnych progów
    results = {}
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        try:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            accuracy = (y_pred == y_test).mean()
            
            print(f"Próg {threshold:.2f}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}, Accuracy = {accuracy:.4f}")
            
            results[float(threshold)] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy)
            }
        except Exception as e:
            print(f"Błąd dla progu {threshold}: {e}")
    
    # Znalezienie najlepszego progu dla F1
    best_threshold = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best_f1 = results[best_threshold]['f1']
    print(f"\nNajlepszy próg dla F1: {best_threshold:.2f} (F1 = {best_f1:.4f})")
    
    # Zapisanie wyników
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_threshold_results.json"), 'w') as f:
        json.dump({
            'model_name': model_name,
            'thresholds': {str(k): v for k, v in results.items()},
            'best_threshold': best_threshold,
            'best_f1': best_f1
        }, f, indent=4)
    
    # Generowanie wykresu Precision-Recall dla różnych progów
    plt.figure(figsize=(10, 6))
    plt.plot(
        [results[t]['recall'] for t in thresholds],
        [results[t]['precision'] for t in thresholds],
        'b-', marker='o'
    )
    
    # Dodanie etykiet z wartościami progów
    for i, t in enumerate(thresholds):
        plt.annotate(
            f"{t:.1f}",
            (results[t]['recall'], results[t]['precision']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Krzywa Precision-Recall dla różnych progów\n{model_name}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_precision_recall_curve.png"))
    
    # Generowanie wykresu F1 dla różnych progów
    plt.figure(figsize=(10, 6))
    plt.plot(
        list(thresholds),
        [results[t]['f1'] for t in thresholds],
        'r-', marker='o'
    )
    plt.xlabel('Próg')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score dla różnych progów\n{model_name}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_f1_thresholds.png"))
    
    # Raport dla najlepszego progu
    y_pred_best = (y_pred_proba >= best_threshold).astype(int)
    report = classification_report(y_test, y_pred_best, output_dict=True)
    
    # Zapisanie raportu dla najlepszego progu
    with open(os.path.join(output_dir, f"{model_name}_best_threshold_report.json"), 'w') as f:
        json.dump(report, f, indent=4)
    
    return results

def analyze_feature_thresholds(model, data, feature, output_dir='.', model_name='model'):
    """Analizuje optymalne progi dla różnych wartości cechy."""
    print(f"\nAnalizowanie optymalnych progów dla cechy '{feature}'...")
    
    # Pobieranie cech i etykiet
    X, y = get_features_and_target(data)
    
    # Sprawdzenie, czy cecha istnieje
    if feature not in X.columns:
        print(f"Cecha '{feature}' nie istnieje w danych!")
        return None
    
    # Unikalne wartości cechy
    unique_values = X[feature].unique()
    if len(unique_values) > 10:  # Dla cech numerycznych ograniczamy liczbę wartości
        print(f"Zbyt wiele unikalnych wartości ({len(unique_values)}), grupowanie...")
        if X[feature].dtype in (np.int64, np.float64):
            X[f'{feature}_binned'] = pd.qcut(X[feature], 10, duplicates='drop')
            feature = f'{feature}_binned'
            unique_values = X[feature].unique()
    
    print(f"Liczba unikalnych wartości cechy '{feature}': {len(unique_values)}")
    
    # Słownik na wyniki
    threshold_results = {}
    
    # Testowanie dla każdej wartości cechy
    for value in unique_values:
        # Filtrujemy dane do wybranej wartości cechy
        value_mask = X[feature] == value
        if sum(value_mask) < 100:  # Pomijamy wartości z małą liczbą przykładów
            continue
        
        X_filtered = X[value_mask]
        y_filtered = y[value_mask]
        
        print(f"\nTestowanie dla {feature} == {value} (liczba przykładów: {sum(value_mask)})")
        
        # Predykcje prawdopodobieństw
        try:
            y_pred_proba = model.predict_proba(X_filtered)[:, 1]
        except Exception as e:
            print(f"Błąd podczas predykcji: {e}")
            continue
        
        # Testujemy różne progi
        thresholds = np.arange(0.1, 1.0, 0.1)
        results = {}
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            try:
                precision = precision_score(y_filtered, y_pred, zero_division=0)
                recall = recall_score(y_filtered, y_pred, zero_division=0)
                f1 = f1_score(y_filtered, y_pred, zero_division=0)
                
                results[float(threshold)] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'count': int(sum(value_mask)),
                    'fraud_rate': float(y_filtered.mean())
                }
            except Exception as e:
                print(f"Błąd dla progu {threshold}: {e}")
        
        # Jeśli mamy wyniki, znajdź najlepszy próg
        if results:
            best_threshold = max(results.items(), key=lambda x: x[1]['f1'])[0]
            best_f1 = results[best_threshold]['f1']
            print(f"Najlepszy próg dla {feature} == {value}: {best_threshold:.2f} (F1 = {best_f1:.4f})")
            
            threshold_results[str(value)] = {
                'optimal_threshold': best_threshold,
                'f1': best_f1,
                'fraud_rate': float(y_filtered.mean()),
                'count': int(sum(value_mask))
            }
    
    # Zapisanie wyników
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_{feature}_thresholds.json"), 'w') as f:
        json.dump(threshold_results, f, indent=4)
    
    # Wizualizacja optymalnych progów
    plt.figure(figsize=(12, 6))
    
    # Dane do wykresu
    values = list(threshold_results.keys())
    thresholds = [threshold_results[v]['optimal_threshold'] for v in values]
    fraud_rates = [threshold_results[v]['fraud_rate'] for v in values]
    
    # Sortowanie po fraud_rate
    sorted_indices = np.argsort(fraud_rates)
    values = [values[i] for i in sorted_indices]
    thresholds = [thresholds[i] for i in sorted_indices]
    fraud_rates = [fraud_rates[i] for i in sorted_indices]
    
    # Wykres zależności progu od fraud_rate
    plt.figure(figsize=(12, 6))
    plt.scatter(fraud_rates, thresholds, alpha=0.7)
    plt.xlabel('Wskaźnik fraudów')
    plt.ylabel('Optymalny próg')
    plt.title(f'Zależność między wskaźnikiem fraudów a optymalnym progiem\nCecha: {feature}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_{feature}_fraud_rate_vs_threshold.png"))
    
    # Wykres optymalnych progów dla różnych wartości
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(values)), thresholds, alpha=0.7)
    plt.xticks(range(len(values)), values, rotation=90)
    plt.xlabel(f'Wartość cechy {feature}')
    plt.ylabel('Optymalny próg')
    plt.title(f'Optymalne progi decyzyjne dla różnych wartości cechy {feature}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_{feature}_optimal_thresholds.png"))
    
    return threshold_results

def main():
    parser = argparse.ArgumentParser(description='Testowanie modelu na wielu transakcjach.')
    parser.add_argument('--model_path', type=str, default='models/improved_smote_model.joblib',
                      help='Ścieżka do modelu (domyślnie: models/improved_smote_model.joblib)')
    parser.add_argument('--data_path', type=str, default='.',
                      help='Ścieżka do katalogu z danymi (domyślnie: .)')
    parser.add_argument('--output_dir', type=str, default='models/batch_test_results',
                      help='Katalog wyjściowy (domyślnie: models/batch_test_results)')
    parser.add_argument('--sample_size', type=int, default=10000,
                      help='Liczba przykładów do testowania (domyślnie: 10000)')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Próg decyzyjny (domyślnie: 0.5)')
    parser.add_argument('--analyze_feature', type=str, default=None,
                      help='Cecha do analizy optymalnych progów (opcjonalnie)')
    parser.add_argument('--model_name', type=str, default='model',
                      help='Nazwa modelu (do użycia w nazwach plików)')
    args = parser.parse_args()
    
    # Wczytanie danych
    users_df, merchants_df, transactions_df = load_data(args.data_path)
    
    # Przygotowanie danych
    data = prepare_data(transactions_df, users_df, merchants_df)
    
    # Dla dużych zbiorów danych, losujemy próbkę
    if args.sample_size < len(data):
        print(f"Losowanie próbki {args.sample_size} przykładów...")
        sample_data = data.sample(args.sample_size, random_state=42)
    else:
        sample_data = data
    
    # Pobieranie cech i etykiet
    X, y = get_features_and_target(sample_data)
    
    # Wczytanie modelu
    model = load_model(args.model_path)
    if model is None:
        return
    
    # Utworzenie katalogu wyjściowego
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Testowanie z różnymi progami
    test_thresholds(model, X, y, output_dir=args.output_dir, model_name=args.model_name)
    
    # Analiza optymalnych progów dla wybranej cechy
    if args.analyze_feature:
        analyze_feature_thresholds(model, sample_data, args.analyze_feature, 
                                  output_dir=args.output_dir, model_name=args.model_name)

if __name__ == "__main__":
    main() 