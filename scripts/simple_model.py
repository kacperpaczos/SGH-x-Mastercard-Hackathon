#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skrypt do trenowania prostego modelu z podstawowymi cechami
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score

# Wczytywanie danych
def load_data(data_path='.'):
    """Wczytuje dane z plików CSV i JSON."""
    print("Wczytywanie danych...")
    
    # Wczytanie danych o użytkownikach
    users_df = pd.read_csv(os.path.join(data_path, 'data/users.csv'))
    print(f"Użytkownicy: {users_df.shape[0]} wierszy")
    
    # Wczytanie danych o sprzedawcach
    merchants_df = pd.read_csv(os.path.join(data_path, 'data/merchants.csv'))
    print(f"Sprzedawcy: {merchants_df.shape[0]} wierszy")
    
    # Wczytanie danych o transakcjach (z pliku JSONL)
    transactions_data = []
    with open(os.path.join(data_path, 'data/transactions.json'), 'r') as f:
        for line in f:
            if line.strip():  # Ignoruj puste linie
                transactions_data.append(json.loads(line))
    
    # Konwersja na DataFrame
    transactions_df = pd.DataFrame(transactions_data)
    print(f"Transakcje: {transactions_df.shape[0]} wierszy")
    
    return users_df, merchants_df, transactions_df

def prepare_data(transactions_df, users_df, merchants_df):
    """Przygotowuje dane do modelu z podstawowymi cechami."""
    print("Przygotowanie danych...")
    
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

def get_features_and_target(data):
    """Zwraca cechy i etykiety."""
    # Podstawowe cechy
    features = [
        'amount', 'channel', 'device', 'payment_method', 
        'is_international', 'session_length_seconds', 
        'hour', 'day_of_week', 'latitude', 'longitude',
        'age', 'sex', 'risk_score',
        'category', 'trust_score', 'has_fraud_history'
    ]
    
    # Sprawdzenie dostępności cech
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"Brakujące cechy: {missing_features}")
        features = [f for f in features if f in data.columns]
    
    print(f"Używane cechy ({len(features)}): {', '.join(features)}")
    
    X = data[features]
    y = data['is_fraud']
    
    return X, y, features

def train_model(X_train, y_train, numerical_features, categorical_features, output_dir='.', model_name='basic_model'):
    """Trenuje model Random Forest."""
    print("Trenowanie modelu...")
    
    # Przetwarzanie cech
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Model Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Trenowanie
    pipeline.fit(X_train, y_train)
    
    # Zapisanie modelu
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model zapisany jako {model_path}")
    
    return pipeline, preprocessor

def evaluate_model(pipeline, X_test, y_test, thresholds=None, output_dir='.', model_name='basic_model'):
    """Ewaluuje model z różnymi progami."""
    print("Ewaluacja modelu...")
    
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    # Predykcje prawdopodobieństw
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Obliczanie miar dla różnych progów
    results = {}
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"Próg {threshold:.1f}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
        
        results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Znalezienie optymalnego progu dla F1
    f1_scores = [results[t]['f1'] for t in thresholds]
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]
    
    print(f"\nNajlepszy próg: {best_threshold:.1f} (F1 = {best_f1:.4f})")
    
    # Używanie najlepszego progu
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # Raport klasyfikacji z optymalnym progiem
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred))
    
    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    
    # Obliczenie ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Macierz pomyłek - wizualizacja
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Prawdziwa klasa')
    plt.title(f'Macierz pomyłek (próg = {best_threshold:.1f})')
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    
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
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"))
    
    # Precision-Recall dla różnych progów
    plt.figure(figsize=(10, 6))
    plt.plot([t for t in thresholds], [results[t]['precision'] for t in thresholds], 'b-', label='Precision')
    plt.plot([t for t in thresholds], [results[t]['recall'] for t in thresholds], 'g-', label='Recall')
    plt.plot([t for t in thresholds], [results[t]['f1'] for t in thresholds], 'r-', label='F1')
    plt.xlabel('Próg')
    plt.ylabel('Wartość')
    plt.title('Metryki według progu')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_precision_recall_threshold.png"))
    
    # Zapisanie wyników
    results_dict = {
        'model_name': model_name,
        'best_threshold': float(best_threshold),
        'best_f1': float(best_f1),
        'threshold_results': {str(k): v for k, v in results.items()},
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(output_dir, f"{model_name}_results.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    return results_dict

def main():
    parser = argparse.ArgumentParser(description='Trenowanie modelu wykrywania fraudów')
    parser.add_argument('--data_path', type=str, default='.',
                      help='Ścieżka do danych')
    parser.add_argument('--output_dir', type=str, default='models/simple_model',
                      help='Katalog wyjściowy')
    parser.add_argument('--sample_size', type=int, default=None,
                      help='Liczba przykładów')
    parser.add_argument('--model_name', type=str, default='basic_model',
                      help='Nazwa modelu')
    args = parser.parse_args()
    
    # Wczytanie danych
    users_df, merchants_df, transactions_df = load_data(args.data_path)
    
    # Przygotowanie danych
    data = prepare_data(transactions_df, users_df, merchants_df)
    
    # Dla dużych zbiorów danych, losujemy próbkę
    if args.sample_size is not None and args.sample_size < len(data):
        print(f"Losowanie próbki {args.sample_size} przykładów...")
        data = data.sample(args.sample_size, random_state=42)
    
    # Pobieranie cech i etykiet
    X, y, features = get_features_and_target(data)
    
    # Podział na cechy kategoryczne i numeryczne
    categorical_features = ['channel', 'device', 'payment_method', 'sex', 'category']
    categorical_features = [f for f in categorical_features if f in features]
    numerical_features = [f for f in features if f not in categorical_features]
    
    print(f"Cechy numeryczne ({len(numerical_features)}): {', '.join(numerical_features)}")
    print(f"Cechy kategoryczne ({len(categorical_features)}): {', '.join(categorical_features)}")
    
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Rozmiar zbioru treningowego: {X_train.shape[0]} przykładów")
    print(f"Rozmiar zbioru testowego: {X_test.shape[0]} przykładów")
    print(f"Procent fraudów w zbiorze treningowym: {100 * y_train.mean():.4f}%")
    print(f"Procent fraudów w zbiorze testowym: {100 * y_test.mean():.4f}%")
    
    # Trenowanie modelu
    pipeline, preprocessor = train_model(
        X_train, y_train, 
        numerical_features, categorical_features,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    
    # Ewaluacja modelu
    evaluate_model(
        pipeline, X_test, y_test,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    
    # Zapisanie metadanych modelu
    model_metadata = {
        'model_name': args.model_name,
        'features': features,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'sample_size': len(data)
    }
    
    with open(os.path.join(args.output_dir, f"{args.model_name}_metadata.json"), 'w') as f:
        json.dump(model_metadata, f, indent=4)
    
    print(f"\nTrening i ewaluacja zakończone! Wyniki zapisane w {args.output_dir}")

if __name__ == "__main__":
    main() 