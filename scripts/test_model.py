#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skrypt do testowania wytrenowanego modelu wykrywania fraudów na pojedynczych transakcjach
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import random
import joblib
from datetime import datetime, timedelta

def load_transaction_example(transaction_file, random_transaction=False, fraud_only=False):
    """Wczytuje przykładową transakcję z pliku JSON.
    
    Parametry:
    ----------
    transaction_file : str
        Ścieżka do pliku z transakcjami
    random_transaction : bool
        Czy wybrać losową transakcję zamiast pierwszej
    fraud_only : bool
        Czy wybierać tylko transakcje fraudowe
    
    Zwraca:
    -------
    transaction : dict
        Słownik z danymi transakcji
    """
    transactions = []
    
    with open(transaction_file, 'r') as f:
        for i, line in enumerate(f):
            if line.strip():
                transaction = json.loads(line)
                if not fraud_only or transaction.get('is_fraud', 0) == 1:
                    transactions.append(transaction)
                if not random_transaction and len(transactions) == 1 and not fraud_only:
                    break
    
    if not transactions:
        raise ValueError("Nie znaleziono transakcji")
    
    if random_transaction:
        return random.choice(transactions)
    else:
        return transactions[0]

def prepare_transaction(transaction, users_df, merchants_df):
    """Przygotowuje pojedynczą transakcję do predykcji.
    
    Parametry:
    ----------
    transaction : dict
        Słownik z danymi transakcji
    users_df : DataFrame
        DataFrame z danymi użytkowników
    merchants_df : DataFrame
        DataFrame z danymi sprzedawców
    
    Zwraca:
    -------
    transaction_df : DataFrame
        DataFrame z pojedynczą transakcją przygotowany do predykcji
    """
    # Konwersja pojedynczej transakcji na DataFrame
    transaction_df = pd.DataFrame([transaction])
    
    # Wyodrębnienie danych o lokalizacji
    transaction_df['latitude'] = transaction_df['location'].apply(lambda x: x['lat'] if isinstance(x, dict) and 'lat' in x else None)
    transaction_df['longitude'] = transaction_df['location'].apply(lambda x: x['long'] if isinstance(x, dict) and 'long' in x else None)
    
    # Przygotowanie danych czasowych
    transaction_df['timestamp'] = pd.to_datetime(transaction_df['timestamp'])
    transaction_df['hour'] = transaction_df['timestamp'].dt.hour
    transaction_df['day_of_week'] = transaction_df['timestamp'].dt.day_name()
    
    # Dodanie danych użytkownika
    user_id = transaction['user_id']
    user_data = users_df[users_df['user_id'] == user_id]
    if not user_data.empty:
        transaction_df['age'] = user_data['age'].values[0]
        transaction_df['sex'] = user_data['sex'].values[0]
        transaction_df['risk_score'] = user_data['risk_score'].values[0]
    else:
        # Jeśli nie znaleziono użytkownika, używamy średnich wartości
        transaction_df['age'] = users_df['age'].mean()
        transaction_df['sex'] = users_df['sex'].mode()[0]
        transaction_df['risk_score'] = users_df['risk_score'].mean()
    
    # Dodanie danych sprzedawcy
    merchant_id = transaction['merchant_id']
    merchant_data = merchants_df[merchants_df['merchant_id'] == merchant_id]
    if not merchant_data.empty:
        transaction_df['category'] = merchant_data['category'].values[0]
        transaction_df['trust_score'] = merchant_data['trust_score'].values[0]
        transaction_df['has_fraud_history'] = merchant_data['has_fraud_history'].values[0]
    else:
        # Jeśli nie znaleziono sprzedawcy, używamy średnich wartości
        transaction_df['category'] = merchants_df['category'].mode()[0]
        transaction_df['trust_score'] = merchants_df['trust_score'].mean()
        transaction_df['has_fraud_history'] = merchants_df['has_fraud_history'].mode()[0]
    
    return transaction_df

def test_model(model, preprocessor, transaction_df, optimal_threshold=0.5):
    """Testuje model na jednej transakcji.
    
    Parametry:
    ----------
    model : obiekt
        Wytrenowany model
    preprocessor : obiekt
        Preprocessor użyty do treningu modelu
    transaction_df : DataFrame
        DataFrame z pojedynczą transakcją
    optimal_threshold : float
        Optymalny próg decyzyjny
    
    Zwraca:
    -------
    results : dict
        Wyniki predykcji
    """
    # Przygotowanie cech
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
    
    # Przygotowanie danych
    X = transaction_df[features]
    
    # Predykcja
    X_preprocessed = preprocessor.transform(X)
    fraud_probability = model.predict_proba(X_preprocessed)[0, 1]
    is_fraud_pred = fraud_probability >= optimal_threshold
    
    # Sprawdzenie faktycznej wartości (jeśli dostępna)
    is_fraud_actual = None
    if 'is_fraud' in transaction_df.columns:
        is_fraud_actual = bool(transaction_df['is_fraud'].values[0])
    
    results = {
        'transaction_id': transaction_df['transaction_id'].values[0],
        'fraud_probability': float(fraud_probability),
        'is_fraud_predicted': bool(is_fraud_pred),
        'threshold_used': optimal_threshold,
        'is_fraud_actual': is_fraud_actual
    }
    
    return results

def create_random_transaction(users_df, merchants_df):
    """Tworzy losową transakcję do testów.
    
    Parametry:
    ----------
    users_df : DataFrame
        DataFrame z danymi użytkowników
    merchants_df : DataFrame
        DataFrame z danymi sprzedawców
    
    Zwraca:
    -------
    transaction : dict
        Słownik z danymi losowej transakcji
    """
    # Wybór losowego użytkownika i sprzedawcy
    user = users_df.sample(1).iloc[0]
    merchant = merchants_df.sample(1).iloc[0]
    
    # Generowanie losowych danych transakcji
    amount = round(random.uniform(1, 500), 2)
    channels = ['online', 'in-store', 'mobile']
    devices = ['Android', 'iOS', 'Web']
    payment_methods = ['credit_card', 'debit_card', 'mobile_payment', 'bank_transfer']
    
    # Losowa lokalizacja w Europie
    lat = random.uniform(35, 60)
    lon = random.uniform(-10, 30)
    
    # Losowy czas w ciągu ostatnich 30 dni
    now = datetime.now()
    random_days = random.randint(0, 30)
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    timestamp = now - timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
    
    # Utworzenie transakcji
    transaction = {
        'transaction_id': f"T{random.randint(10000, 99999)}",
        'timestamp': timestamp.isoformat(),
        'user_id': user['user_id'],
        'merchant_id': merchant['merchant_id'],
        'amount': amount,
        'channel': random.choice(channels),
        'currency': 'EUR',
        'device': random.choice(devices),
        'location': {'lat': lat, 'long': lon},
        'payment_method': random.choice(payment_methods),
        'is_international': random.random() < 0.2,
        'session_length_seconds': random.randint(30, 600),
        'is_first_time_merchant': random.random() < 0.3
    }
    
    return transaction

def main():
    parser = argparse.ArgumentParser(description='Test modelu na pojedynczej transakcji')
    parser.add_argument('--model_path', type=str, default='models/improved_smote_model.joblib',
                      help='Ścieżka do modelu')
    parser.add_argument('--preprocessor_path', type=str, default='models/improved_smote_preprocessor.joblib',
                      help='Ścieżka do preprocessora')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Próg decyzyjny')
    parser.add_argument('--data_path', type=str, default='.',
                      help='Ścieżka do danych')
    parser.add_argument('--random', action='store_true',
                      help='Losowa transakcja')
    parser.add_argument('--fraud_only', action='store_true',
                      help='Tylko fraudy')
    parser.add_argument('--generate', action='store_true',
                      help='Generuj transakcję')
    args = parser.parse_args()
    
    print("Wczytywanie modelu i preprocessora...")
    model = joblib.load(args.model_path)
    preprocessor = joblib.load(args.preprocessor_path)
    
    print("Wczytywanie danych...")
    users_df = pd.read_csv(os.path.join(args.data_path, 'users.csv'))
    merchants_df = pd.read_csv(os.path.join(args.data_path, 'merchants.csv'))
    
    if args.generate:
        print("Generowanie losowej transakcji...")
        transaction = create_random_transaction(users_df, merchants_df)
    else:
        print("Wczytywanie transakcji...")
        transaction = load_transaction_example(
            os.path.join(args.data_path, 'transactions.json'),
            random_transaction=args.random,
            fraud_only=args.fraud_only
        )
    
    print("\nTransakcja:")
    print(json.dumps(transaction, indent=2))
    
    print("\nPrzygotowanie transakcji...")
    transaction_df = prepare_transaction(transaction, users_df, merchants_df)
    
    print("\nTestowanie modelu...")
    results = test_model(model, preprocessor, transaction_df, args.threshold)
    
    print("\nWyniki:")
    print(f"ID transakcji: {results['transaction_id']}")
    print(f"Prawdopodobieństwo fraudu: {results['fraud_probability']:.4f}")
    print(f"Przewidywany fraud: {'Tak' if results['is_fraud_predicted'] else 'Nie'}")
    print(f"Użyty próg: {results['threshold_used']}")
    
    if results['is_fraud_actual'] is not None:
        print(f"Faktyczny fraud: {'Tak' if results['is_fraud_actual'] else 'Nie'}")
        print(f"Poprawna predykcja: {'Tak' if results['is_fraud_predicted'] == results['is_fraud_actual'] else 'Nie'}")

if __name__ == "__main__":
    main() 