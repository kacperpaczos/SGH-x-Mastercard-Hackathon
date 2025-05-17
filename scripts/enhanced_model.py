#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skrypt do trenowania ulepszonych modeli detekcji fraudów:
- Rozszerzenie o zaawansowane cechy czasowe
- Testowanie różnych algorytmów (Random Forest, XGBoost, LightGBM)
- Dostosowanie progów decyzyjnych dla różnych kombinacji cech
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
from datetime import datetime, timedelta
from collections import defaultdict

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score

# Próba importu dodatkowych algorytmów - opcjonalne
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost nie jest dostępny. Zainstaluj pakiet xgboost dla dodatkowych modeli.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM nie jest dostępny. Zainstaluj pakiet lightgbm dla dodatkowych modeli.")


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
    """Konwertuje kolumnę timestamp na format datetime i dodaje rozszerzone cechy czasowe."""
    print("Rozszerzanie cech czasowych...")
    
    # Podstawowe cechy czasowe
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['day_of_week_num'] = df['timestamp'].dt.dayofweek
    
    # Zaawansowane cechy czasowe
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['day'] = df['timestamp'].dt.day
    df['minute'] = df['timestamp'].dt.minute
    
    # Pochodne cechy czasowe
    df['is_weekend'] = df['day_of_week_num'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_night'] = df['hour'].apply(lambda x: 1 if (x >= 22 or x < 6) else 0)
    df['is_evening'] = df['hour'].apply(lambda x: 1 if (x >= 18 and x < 22) else 0)
    df['time_of_day'] = df['hour'].apply(lambda x: 'night' if (x >= 22 or x < 6) 
                                       else 'morning' if (x >= 6 and x < 12)
                                       else 'afternoon' if (x >= 12 and x < 18)
                                       else 'evening')
    
    # Kwartał roku
    df['quarter'] = df['timestamp'].dt.quarter
    
    return df

def extract_location_features(df):
    """Wyodrębnić cechy lokalizacyjne i dodać nowe cechy na ich podstawie."""
    print("Przetwarzanie danych lokalizacyjnych...")
    
    # Wyodrębnienie danych o lokalizacji
    df['latitude'] = df['location'].apply(lambda x: x['lat'] if isinstance(x, dict) and 'lat' in x else None)
    df['longitude'] = df['location'].apply(lambda x: x['long'] if isinstance(x, dict) and 'long' in x else None)
    
    # Grupowanie lokalizacji na regiony (uproszczone - w rzeczywistej aplikacji użylibyśmy lepszej metody)
    df['region_lat'] = df['latitude'].apply(lambda x: round(x / 10) * 10 if x is not None else None)
    df['region_long'] = df['longitude'].apply(lambda x: round(x / 10) * 10 if x is not None else None)
    df['region'] = df.apply(lambda row: f"{row['region_lat']}_{row['region_long']}" if row['region_lat'] is not None else "unknown", axis=1)
    
    return df

def create_user_aggregates(merged_df):
    """Tworzenie zagregowanych cech dla użytkowników na podstawie ich historii transakcji."""
    print("Tworzenie cech zagregowanych dla użytkowników...")
    
    # Sortowanie według użytkownika i czasu
    merged_df = merged_df.sort_values(['user_id', 'timestamp'])
    
    # Grupowanie według użytkownika
    user_groups = merged_df.groupby('user_id')
    
    # Tworzenie nowych DataFrame dla przechowywania zagregowanych cech
    user_aggs = pd.DataFrame(index=merged_df.index)
    
    # Średnia kwota transakcji użytkownika
    user_avg_amount = user_groups['amount'].transform('mean')
    user_aggs['user_avg_amount'] = user_avg_amount
    
    # Stosunek kwoty transakcji do średniej użytkownika
    user_aggs['amount_to_avg_ratio'] = merged_df['amount'] / user_avg_amount
    
    # Liczba transakcji użytkownika
    user_aggs['user_transaction_count'] = user_groups['transaction_id'].transform('count')
    
    # Średni odstęp między transakcjami
    def calculate_time_diffs(group):
        if len(group) <= 1:
            return pd.Series([0] * len(group))
        time_diffs = group['timestamp'].diff().dt.total_seconds() / 3600  # w godzinach
        time_diffs.iloc[0] = 0
        return time_diffs
    
    user_aggs['hours_since_last_transaction'] = user_groups.apply(calculate_time_diffs).reset_index(level=0, drop=True)
    
    # Liczba różnych urządzeń użytkownika
    user_aggs['user_device_count'] = user_groups['device'].transform('nunique')
    
    # Liczba różnych sprzedawców dla użytkownika
    user_aggs['user_merchant_count'] = user_groups['merchant_id'].transform('nunique')
    
    # Procent transakcji międzynarodowych użytkownika
    user_aggs['user_international_ratio'] = user_groups['is_international'].transform('mean')
    
    # Liczba różnych metod płatności
    user_aggs['user_payment_method_count'] = user_groups['payment_method'].transform('nunique')
    
    # Łączenie z oryginalnym DataFrame
    enhanced_df = pd.concat([merged_df, user_aggs], axis=1)
    
    return enhanced_df

def create_merchant_aggregates(merged_df):
    """Tworzenie zagregowanych cech dla sprzedawców na podstawie historii transakcji."""
    print("Tworzenie cech zagregowanych dla sprzedawców...")
    
    # Grupowanie według sprzedawcy
    merchant_groups = merged_df.groupby('merchant_id')
    
    # Tworzenie nowych DataFrame dla przechowywania zagregowanych cech
    merchant_aggs = pd.DataFrame(index=merged_df.index)
    
    # Średnia kwota transakcji sprzedawcy
    merchant_aggs['merchant_avg_amount'] = merchant_groups['amount'].transform('mean')
    
    # Stosunek kwoty transakcji do średniej sprzedawcy
    merchant_aggs['amount_to_merchant_avg_ratio'] = merged_df['amount'] / merchant_aggs['merchant_avg_amount']
    
    # Liczba transakcji sprzedawcy
    merchant_aggs['merchant_transaction_count'] = merchant_groups['transaction_id'].transform('count')
    
    # Liczba unikalnych użytkowników dla sprzedawcy
    merchant_aggs['merchant_user_count'] = merchant_groups['user_id'].transform('nunique')
    
    # Procent transakcji międzynarodowych sprzedawcy
    merchant_aggs['merchant_international_ratio'] = merchant_groups['is_international'].transform('mean')
    
    # Łączenie z oryginalnym DataFrame
    enhanced_df = pd.concat([merged_df, merchant_aggs], axis=1)
    
    return enhanced_df

def prepare_data(transactions_df, users_df, merchants_df, create_advanced_features=True):
    """Przygotowuje dane do treningu modelu z zaawansowanymi cechami."""
    print("Przygotowanie danych z rozszerzonymi cechami...")
    
    # Przetwarzanie czasu
    transactions_df = parse_timestamp(transactions_df)
    
    # Przetwarzanie lokalizacji
    transactions_df = extract_location_features(transactions_df)
    
    # Łączenie danych z użytkownikami i sprzedawcami
    merged_data = transactions_df.merge(users_df[['user_id', 'age', 'sex', 'risk_score', 'country']], 
                                      on='user_id', how='left')
    merged_data = merged_data.merge(merchants_df[['merchant_id', 'category', 'trust_score', 'has_fraud_history', 'country', 'number_of_alerts_last_6_months']], 
                                   on='merchant_id', how='left')
    
    # Dodanie cechy wskazującej, czy kraj użytkownika jest taki sam jak kraj sprzedawcy
    merged_data['same_country'] = (merged_data['country_x'] == merged_data['country_y']).astype(int)
    merged_data.rename(columns={'country_x': 'user_country', 'country_y': 'merchant_country'}, inplace=True)
    
    if create_advanced_features:
        # Tworzenie zagregowanych cech dla użytkowników
        merged_data = create_user_aggregates(merged_data)
        
        # Tworzenie zagregowanych cech dla sprzedawców
        merged_data = create_merchant_aggregates(merged_data)
        
        # Interakcje między cechami
        merged_data['risk_trust_product'] = merged_data['risk_score'] * merged_data['trust_score']
        merged_data['amount_session_ratio'] = merged_data['amount'] / (merged_data['session_length_seconds'] + 1)
    
    return merged_data

def get_feature_list(include_advanced=True):
    """Zwraca listę cech do wykorzystania w modelu."""
    
    # Podstawowe cechy
    basic_features = [
        # Cechy transakcji
        'amount', 'channel', 'device', 'payment_method', 
        'is_international', 'session_length_seconds', 'is_first_time_merchant',
        'hour', 'day_of_week', 
        
        # Cechy użytkownika
        'age', 'sex', 'risk_score',
        
        # Cechy sprzedawcy
        'category', 'trust_score', 'has_fraud_history', 'number_of_alerts_last_6_months',
        
        # Cechy lokalizacji
        'latitude', 'longitude',
        
        # Relacja użytkownik-sprzedawca
        'same_country'
    ]
    
    # Dodatkowe zaawansowane cechy czasowe
    time_features = [
        'day_of_week_num', 'month', 'day', 'minute',
        'is_weekend', 'is_night', 'is_evening', 'time_of_day', 'quarter'
    ]
    
    # Cechy zagregowane
    aggregated_features = [
        'user_avg_amount', 'amount_to_avg_ratio', 'user_transaction_count', 
        'hours_since_last_transaction', 'user_device_count', 'user_merchant_count',
        'user_international_ratio', 'user_payment_method_count',
        'merchant_avg_amount', 'amount_to_merchant_avg_ratio', 'merchant_transaction_count',
        'merchant_user_count', 'merchant_international_ratio',
        'risk_trust_product', 'amount_session_ratio'
    ]
    
    if include_advanced:
        return basic_features + time_features + aggregated_features
    else:
        return basic_features

def get_categorical_features(feature_list):
    """Zwraca listę cech kategorialnych z podanej listy cech."""
    categorical_features = [
        'channel', 'device', 'payment_method', 'day_of_week', 'sex', 'category',
        'time_of_day', 'user_country', 'merchant_country', 'region'
    ]
    
    # Zwracamy tylko te cechy kategorialne, które są w podanej liście cech
    return [f for f in categorical_features if f in feature_list]

def create_model(model_type='randomforest', params=None):
    """Tworzy model wskazanego typu z opcjonalnymi parametrami."""
    
    if params is None:
        params = {}
    
    if model_type.lower() == 'randomforest':
        default_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        # Aktualizacja domyślnych parametrów
        default_params.update(params)
        return RandomForestClassifier(**default_params)
    
    elif model_type.lower() == 'xgboost' and XGBOOST_AVAILABLE:
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'scale_pos_weight': 10,  # Dla nierównowagi klas
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        return xgb.XGBClassifier(**default_params)
    
    elif model_type.lower() == 'lightgbm' and LIGHTGBM_AVAILABLE:
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        return lgb.LGBMClassifier(**default_params)
    
    else:
        print(f"Model {model_type} nie jest obsługiwany lub biblioteka nie jest zainstalowana. Używam RandomForest.")
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

def train_and_evaluate_model(data, model_type='randomforest', use_advanced_features=True, model_params=None, output_dir='models'):
    """Trenuje i ewaluuje model określonego typu z zaawansowanymi cechami."""
    
    # Wybór cech do modelu
    features = get_feature_list(include_advanced=use_advanced_features)
    categorical_features = get_categorical_features(features)
    numerical_features = [f for f in features if f not in categorical_features]
    
    print(f"Używane cechy: {len(features)} cech ({len(numerical_features)} numerycznych, {len(categorical_features)} kategorialnych)")
    
    # Przygotowanie danych
    X = data[features]
    y = data['is_fraud']
    
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Rozmiar zbioru treningowego: {X_train.shape[0]} przykładów")
    print(f"Rozmiar zbioru testowego: {X_test.shape[0]} przykładów")
    print(f"Procent fraudów w zbiorze treningowym: {100 * y_train.mean():.4f}%")
    print(f"Procent fraudów w zbiorze testowym: {100 * y_test.mean():.4f}%")
    
    # Przetwarzanie cech
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Usuń kolumny, które nie są w numerical_features ani categorical_features
    )
    
    # Utworzenie modelu
    model = create_model(model_type, model_params)
    
    # Utworzenie potoku
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Trenowanie modelu
    print(f"\nTrening modelu {model_type}...")
    pipeline.fit(X_train, y_train)
    
    # Predykcje
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Obliczanie miar dla różnych progów decyzyjnych
    results = {}
    thresholds = np.arange(0.1, 0.91, 0.1)
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Próg {threshold:.1f}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
        
        results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Znajdowanie optymalnego progu dla F1
    f1_scores = [results[t]['f1'] for t in thresholds]
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]
    
    print(f"\nNajlepszy próg dla F1: {best_threshold:.1f} (F1 = {best_f1:.4f})")
    
    # Używanie najlepszego progu
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # Raport klasyfikacji z optymalnym progiem
    print("\n--- Raport klasyfikacji z optymalnym progiem ---")
    print(classification_report(y_test, y_pred))
    
    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    
    # Obliczenie ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Zapisanie modelu
    os.makedirs(output_dir, exist_ok=True)
    model_name = f"{model_type}_{'advanced' if use_advanced_features else 'basic'}"
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    preprocessor_path = os.path.join(output_dir, f"{model_name}_preprocessor.joblib")
    
    joblib.dump(pipeline, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"Model zapisany jako {model_path}")
    
    # Macierz pomyłek - wizualizacja
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Prawdziwa klasa')
    plt.title(f'Macierz pomyłek ({model_type}, próg = {best_threshold:.1f})')
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    
    # Krzywa ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Krzywa ROC ({model_type})')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"))
    
    # Precision-Recall dla różnych progów
    plt.figure(figsize=(10, 6))
    plt.plot([t for t in thresholds], [results[t]['precision'] for t in thresholds], 'b-', label='Precision')
    plt.plot([t for t in thresholds], [results[t]['recall'] for t in thresholds], 'g-', label='Recall')
    plt.plot([t for t in thresholds], [results[t]['f1'] for t in thresholds], 'r-', label='F1')
    plt.xlabel('Próg')
    plt.ylabel('Wartość')
    plt.title(f'Precision, Recall i F1 dla różnych progów ({model_type})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_precision_recall_threshold.png"))
    
    # Zapisanie wyników
    results_dict = {
        'model_type': model_type,
        'use_advanced_features': use_advanced_features,
        'best_threshold': float(best_threshold),
        'best_f1': float(best_f1),
        'threshold_results': {str(k): v for k, v in results.items()},
        'feature_count': len(features),
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(output_dir, f"{model_name}_results.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    return pipeline, preprocessor, results_dict

def find_optimal_thresholds_by_feature(data, model, preprocessor, feature_name, output_dir='models'):
    """Znajduje optymalne progi decyzyjne dla różnych wartości wybranej cechy."""
    print(f"\nZnajdowanie optymalnych progów dla cechy '{feature_name}'...")
    
    # Wybór wszystkich cech
    features = get_feature_list(include_advanced=True)
    categorical_features = get_categorical_features(features)
    numerical_features = [f for f in features if f not in categorical_features]
    
    # Przygotowanie danych
    X = data[features]
    y = data['is_fraud']
    
    # Podział na zbiór treningowy i testowy
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Dodajemy wartość wybranej cechy do wyniku
    feature_values = X_test[feature_name].unique()
    results = {}
    
    # Predykcje
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Dla każdej wartości cechy, znajdź optymalny próg
    for value in feature_values:
        # Wybierz indeksy przykładów o danej wartości cechy
        value_indices = X_test[feature_name] == value
        
        if sum(value_indices) < 100:  # Pomijamy wartości z małą liczbą przykładów
            continue
        
        # Filtrujemy dane testowe do wybranej wartości cechy
        y_test_filtered = y_test[value_indices]
        y_proba_filtered = y_pred_proba[value_indices]
        
        # Jeśli wszystkie przykłady mają tę samą etykietę, nie możemy znaleźć optymalnego progu
        if y_test_filtered.nunique() < 2:
            continue
        
        # Obliczamy różne miary dla różnych progów
        thresholds = np.arange(0.1, 0.91, 0.1)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba_filtered >= threshold).astype(int)
            f1 = f1_score(y_test_filtered, y_pred)
            f1_scores.append(f1)
        
        # Znajdź optymalny próg
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        # Zapisz wynik
        results[value] = {
            'optimal_threshold': float(best_threshold),
            'f1_score': float(best_f1),
            'sample_count': int(sum(value_indices))
        }
    
    # Zapisanie wyników do pliku
    with open(os.path.join(output_dir, f"optimal_thresholds_by_{feature_name}.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Wizualizacja optymalnych progów dla wartości cechy
    plt.figure(figsize=(12, 6))
    values = list(results.keys())
    thresholds = [results[v]['optimal_threshold'] for v in values]
    
    # Jeśli cecha jest kategorialna, konwertuj wartości na stringi
    if feature_name in categorical_features:
        values = [str(v) for v in values]
    
    # Jeśli liczba unikalnych wartości jest duża, pokaż tylko niektóre
    if len(values) > 15:
        step = len(values) // 15 + 1
        values = values[::step]
        thresholds = thresholds[::step]
    
    plt.bar(range(len(values)), thresholds)
    plt.xticks(range(len(values)), values, rotation=45, ha='right')
    plt.xlabel(f'Wartość cechy {feature_name}')
    plt.ylabel('Optymalny próg')
    plt.title(f'Optymalne progi decyzyjne dla różnych wartości cechy {feature_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"optimal_thresholds_by_{feature_name}.png"))
    
    return results

def apply_adaptive_thresholds(X_test, y_test, y_pred_proba, thresholds_by_features):
    """Stosuje adaptacyjne progi decyzyjne na podstawie wartości cech."""
    
    # Inicjalizacja domyślnym progiem
    default_threshold = 0.5
    y_pred = np.zeros(len(y_test), dtype=int)
    
    # Dla każdego przykładu, sprawdź wartości cech i zastosuj odpowiedni próg
    for i in range(len(X_test)):
        applied_threshold = default_threshold
        
        # Sprawdź, czy możemy zastosować adaptacyjny próg dla jakiejś cechy
        for feature_name, thresholds in thresholds_by_features.items():
            feature_value = X_test.iloc[i][feature_name]
            
            # Sprawdź, czy mamy optymalny próg dla tej wartości cechy
            if feature_value in thresholds:
                applied_threshold = thresholds[feature_value]['optimal_threshold']
                break
        
        # Zastosuj wybrany próg
        y_pred[i] = 1 if y_pred_proba[i] >= applied_threshold else 0
    
    # Ocena wyników
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- Wyniki z adaptacyjnymi progami ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return y_pred, {'precision': precision, 'recall': recall, 'f1': f1}

def main():
    parser = argparse.ArgumentParser(description='Trenowanie zaawansowanych modeli detekcji fraudów.')
    parser.add_argument('--model_type', type=str, default='randomforest', choices=['randomforest', 'xgboost', 'lightgbm'],
                        help='Typ modelu do treningu (domyślnie: randomforest)')
    parser.add_argument('--no_advanced_features', action='store_true',
                        help='Nie używaj zaawansowanych cech')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Katalog wyjściowy (domyślnie: models)')
    parser.add_argument('--data_path', type=str, default='.',
                        help='Ścieżka do katalogu z danymi (domyślnie: .)')
    parser.add_argument('--adaptive_thresholds', action='store_true',
                        help='Testuj adaptacyjne progi dla różnych cech')
    parser.add_argument('--feature_for_threshold', type=str, default='channel',
                        help='Cecha dla której będą testowane adaptacyjne progi (domyślnie: channel)')
    args = parser.parse_args()
    
    # Wczytanie danych
    users_df, merchants_df, transactions_df = load_data(args.data_path)
    
    # Przygotowanie danych z zaawansowanymi cechami
    data = prepare_data(transactions_df, users_df, merchants_df, create_advanced_features=not args.no_advanced_features)
    
    # Trenowanie modelu
    model, preprocessor, results = train_and_evaluate_model(
        data, 
        model_type=args.model_type,
        use_advanced_features=not args.no_advanced_features,
        output_dir=args.output_dir
    )
    
    # Jeśli wybrano testowanie adaptacyjnych progów
    if args.adaptive_thresholds:
        # Znajdź optymalne progi dla wybranej cechy
        feature_thresholds = find_optimal_thresholds_by_feature(
            data, 
            model, 
            preprocessor,
            args.feature_for_threshold,
            output_dir=args.output_dir
        )
        
        # Przygotowanie danych do testowania adaptacyjnych progów
        features = get_feature_list(include_advanced=not args.no_advanced_features)
        X = data[features]
        y = data['is_fraud']
        
        # Podział na zbiór treningowy i testowy
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Predykcje prawdopodobieństw
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Zastosowanie adaptacyjnych progów
        thresholds_by_features = {args.feature_for_threshold: feature_thresholds}
        adaptive_y_pred, adaptive_results = apply_adaptive_thresholds(
            X_test, y_test, y_pred_proba, thresholds_by_features
        )
        
        # Zapisanie wyników adaptacyjnych progów
        with open(os.path.join(args.output_dir, f"adaptive_thresholds_results_{args.model_type}.json"), 'w') as f:
            json.dump({
                'feature': args.feature_for_threshold,
                'adaptive_results': adaptive_results,
                'standard_results': {
                    'precision': results['threshold_results'][str(results['best_threshold'])]['precision'],
                    'recall': results['threshold_results'][str(results['best_threshold'])]['recall'],
                    'f1': results['threshold_results'][str(results['best_threshold'])]['f1']
                }
            }, f, indent=4)
    
    print("\nTrening i ewaluacja zakończone!")

if __name__ == "__main__":
    main() 