#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automatyczny skrypt do budowania i testowania modelu wykrywania fraudów
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Dodajemy folder scripts do ścieżki Pythona, jeśli uruchamiamy z innego katalogu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def prepare_data(users_df, merchants_df, transactions_df):
    """Przygotowuje dane do treningu modelu."""
    print("Przygotowanie danych do modelu...")
    
    # Wyodrębnienie danych o lokalizacji
    transactions_df['latitude'] = transactions_df['location'].apply(lambda x: x['lat'] if isinstance(x, dict) and 'lat' in x else None)
    transactions_df['longitude'] = transactions_df['location'].apply(lambda x: x['long'] if isinstance(x, dict) and 'long' in x else None)
    
    # Łączenie danych z użytkownikami i sprzedawcami
    merged_data = transactions_df.merge(users_df[['user_id', 'age', 'sex', 'risk_score']], 
                                        on='user_id', how='left')
    merged_data = merged_data.merge(merchants_df[['merchant_id', 'category', 'trust_score', 'has_fraud_history']], 
                                   on='merchant_id', how='left')
    
    return merged_data

def train_model(data, model_type='basic', use_smote=True, output_dir='models'):
    """Trenuje model detekcji fraudów.
    
    Parametry:
    ----------
    data : DataFrame
        Dane przygotowane do treningu modelu
    model_type : str
        Typ modelu: 'basic' (podstawowy) lub 'improved' (ulepszony)
    use_smote : bool
        Czy używać SMOTE do zbilansowania klas
    output_dir : str
        Katalog wyjściowy do zapisania modelu i wykresów
    
    Zwraca:
    -------
    model : obiekt
        Wytrenowany model
    results : dict
        Wyniki ewaluacji modelu
    """
    
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
    
    # Definiowanie cech kategorialnych i numerycznych
    categorical_features = [
        'channel', 'device', 'payment_method', 'day_of_week', 'sex', 'category'
    ]
    numerical_features = [f for f in features if f not in categorical_features]
    
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
        ])
    
    # Zastosowanie preprocessora
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Zastosowanie SMOTE do zbilansowania klas w zbiorze treningowym, jeśli wybrano
    if use_smote:
        print("Zastosowanie SMOTE do zbilansowania klas...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
        
        print(f"Rozmiar zbioru treningowego po SMOTE: {X_train_resampled.shape[0]} przykładów")
        print(f"Procent fraudów w zbiorze treningowym po SMOTE: {100 * y_train_resampled.mean():.4f}%")
    else:
        X_train_resampled, y_train_resampled = X_train_preprocessed, y_train
    
    # Wybór modelu w zależności od parametru model_type
    if model_type == 'improved':
        print("Trening ulepszonego modelu Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    else:  # basic
        print("Trening podstawowego modelu Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    
    # Trenowanie modelu
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predykcje
    y_pred = model.predict(X_test_preprocessed)
    y_prob = model.predict_proba(X_test_preprocessed)[:, 1]
    
    # Ewaluacja modelu
    print("\n--- Raport klasyfikacji ---")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    # Tworzenie katalogu wyjściowego, jeśli nie istnieje
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{model_type}_smote" if use_smote else f"{model_type}_basic"
    
    # Zapisanie modelu
    model_filepath = os.path.join(output_dir, f"{prefix}_model.joblib")
    preprocessor_filepath = os.path.join(output_dir, f"{prefix}_preprocessor.joblib")
    
    joblib.dump(model, model_filepath)
    joblib.dump(preprocessor, preprocessor_filepath)
    print(f"Model zapisany w {model_filepath}")
    print(f"Preprocessor zapisany w {preprocessor_filepath}")
    
    # Przygotowanie ścieżek do wykresów
    cm_filepath = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
    roc_filepath = os.path.join(output_dir, f"{prefix}_roc_curve.png")
    pr_filepath = os.path.join(output_dir, f"{prefix}_pr_curve.png")
    importance_filepath = os.path.join(output_dir, f"{prefix}_feature_importance.png")
    
    # Macierz pomyłek
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Prawdziwa klasa')
    plt.title('Macierz pomyłek')
    plt.savefig(cm_filepath)
    
    # Krzywa ROC
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywa ROC')
    plt.legend(loc="lower right")
    plt.savefig(roc_filepath)
    
    # Krzywa Precision-Recall
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Krzywa Precision-Recall')
    plt.legend(loc="upper right")
    plt.savefig(pr_filepath)
    
    # Ważność cech
    if hasattr(model, 'feature_importances_'):
        # Zbieramy nazwy cech po przekształceniu
        feature_names = []
        
        # Cechy numeryczne (nie zmieniają nazw)
        for name in numerical_features:
            feature_names.append(name)
        
        # Cechy kategoryczne po one-hot encoding
        if hasattr(preprocessor.transformers_[1][1], 'get_feature_names_out'):
            cat_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)
            feature_names.extend(cat_feature_names)
        
        # Ważność cech
        importances = model.feature_importances_
        
        if len(importances) == len(feature_names):
            # Sortowanie cech według ważności
            indices = np.argsort(importances)[::-1]
            
            # Wydrukowanie 15 najważniejszych cech
            print("\n--- Top 15 najważniejszych cech ---")
            for i, idx in enumerate(indices[:15]):
                print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
            
            # Wykres ważności cech
            plt.figure(figsize=(12, 8))
            plt.title("Ważność cech")
            plt.bar(range(min(15, len(indices))), importances[indices[:15]], align="center")
            plt.xticks(range(min(15, len(indices))), [feature_names[i] for i in indices[:15]], rotation=90)
            plt.tight_layout()
            plt.savefig(importance_filepath)
    
    # Optymalizacja progu decyzyjnego
    thresholds = np.arange(0, 1, 0.05)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        report_thresh = classification_report(y_test, y_pred_thresh, output_dict=True)
        f1 = report_thresh['1']['f1-score']
        precision = report_thresh['1']['precision']
        recall = report_thresh['1']['recall']
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # Znalezienie optymalnego progu
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    print(f"\nOptymalny próg decyzyjny: {optimal_threshold:.2f} (F1-score: {optimal_f1:.4f})")
    
    # Zastosowanie optymalnego progu
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    print("\n--- Raport klasyfikacji z optymalnym progiem ---")
    optimal_report = classification_report(y_test, y_pred_optimal, output_dict=True)
    print(classification_report(y_test, y_pred_optimal))
    
    # Przygotowanie rezultatów
    results = {
        'model_type': model_type,
        'use_smote': use_smote,
        'basic_metrics': report,
        'optimal_threshold': optimal_threshold,
        'optimal_metrics': optimal_report,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'feature_importance': list(zip(feature_names, model.feature_importances_)) if hasattr(model, 'feature_importances_') else None
    }
    
    return model, preprocessor, results

def predict_transaction(transaction_data, model, preprocessor, optimal_threshold=0.5):
    """Przewiduje, czy transakcja jest fraudem.
    
    Parametry:
    ----------
    transaction_data : dict
        Dane transakcji
    model : obiekt
        Wytrenowany model
    preprocessor : obiekt
        Preprocessor użyty do treningu modelu
    optimal_threshold : float
        Optymalny próg decyzyjny
    
    Zwraca:
    -------
    prediction : dict
        Wynik predykcji z prawdopodobieństwem i decyzją
    """
    # Konwersja pojedynczej transakcji na DataFrame
    transaction_df = pd.DataFrame([transaction_data])
    
    # Przygotowanie cech
    X = preprocessor.transform(transaction_df)
    
    # Predykcja
    fraud_probability = model.predict_proba(X)[0, 1]
    is_fraud = fraud_probability >= optimal_threshold
    
    prediction = {
        'fraud_probability': float(fraud_probability),
        'is_fraud': bool(is_fraud),
        'threshold_used': optimal_threshold
    }
    
    return prediction

def main():
    parser = argparse.ArgumentParser(description='Automatyczny skrypt do budowania i testowania modelu wykrywania fraudów')
    parser.add_argument('--model_type', choices=['basic', 'improved'], default='improved',
                        help='Typ modelu do treningu: basic lub improved')
    parser.add_argument('--no_smote', action='store_true',
                        help='Nie używaj SMOTE do zbilansowania klas')
    parser.add_argument('--output_dir', default='models',
                        help='Katalog wyjściowy do zapisania modelu i wykresów')
    parser.add_argument('--data_path', default='.',
                        help='Ścieżka do katalogu z danymi')
    
    args = parser.parse_args()
    
    # Wczytanie danych
    users_df, merchants_df, transactions_df = load_data(args.data_path)
    
    # Przetwarzanie timestampów
    transactions_df = parse_timestamp(transactions_df)
    
    # Przygotowanie danych
    merged_data = prepare_data(users_df, merchants_df, transactions_df)
    
    # Trenowanie modelu
    model, preprocessor, results = train_model(
        data=merged_data,
        model_type=args.model_type,
        use_smote=not args.no_smote,
        output_dir=args.output_dir
    )
    
    # Zapisanie wyników do pliku JSON
    results_filepath = os.path.join(args.output_dir, f"{args.model_type}_{'smote' if not args.no_smote else 'basic'}_results.json")
    
    # Konwersja numpy types na natywne typy Pythona
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    # Uproszczenie wyników do zapisu
    serializable_results = {
        'model_type': results['model_type'],
        'use_smote': results['use_smote'],
        'basic_metrics': {
            'precision': results['basic_metrics']['1']['precision'],
            'recall': results['basic_metrics']['1']['recall'],
            'f1_score': results['basic_metrics']['1']['f1-score']
        },
        'optimal_threshold': results['optimal_threshold'],
        'optimal_metrics': {
            'precision': results['optimal_metrics']['1']['precision'],
            'recall': results['optimal_metrics']['1']['recall'],
            'f1_score': results['optimal_metrics']['1']['f1-score']
        },
        'roc_auc': results['roc_auc'],
        'avg_precision': results['avg_precision']
    }
    
    # Dodanie informacji o ważności cech (top 15)
    if results['feature_importance']:
        top_features = sorted(results['feature_importance'], key=lambda x: x[1], reverse=True)[:15]
        serializable_results['top_features'] = [{'name': name, 'importance': float(importance)} 
                                               for name, importance in top_features]
    
    # Zapisanie wyników
    with open(results_filepath, 'w') as f:
        json.dump(convert_to_serializable(serializable_results), f, indent=4)
    
    print(f"\nAnaliza modelu {args.model_type} zakończona!")
    print(f"Wyniki zapisane w {results_filepath}")

if __name__ == "__main__":
    main() 