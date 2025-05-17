#!/usr/bin/env python
# -*- coding: utf-8 -*-

from explore_data import load_data
from analyze_data import parse_timestamp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython import embed

print("Wczytywanie danych...")
users_df, merchants_df, transactions_df = load_data()

print("Przetwarzanie danych czasowych...")
transactions_df = parse_timestamp(transactions_df)

print(f"\nUżytkownicy: {len(users_df)}")
print(f"Sprzedawcy: {len(merchants_df)}")
print(f"Transakcje: {len(transactions_df)}")
print(f"Fraudy: {100 * transactions_df['is_fraud'].mean():.4f}%")

def plot_fraud_by_feature(feature, title=None, sort=True):
    if title is None:
        title = f'Fraudy według {feature}'
    
    fraud_by_feature = transactions_df.groupby(feature)['is_fraud'].mean()
    if sort:
        fraud_by_feature = fraud_by_feature.sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    fraud_by_feature.plot(kind='bar')
    plt.title(title)
    plt.ylabel('Procent')
    plt.xlabel(feature)
    plt.tight_layout()
    plt.show()

def analyze_user(user_id):
    user = users_df[users_df['user_id'] == user_id]
    if user.empty:
        print(f"Nie znaleziono użytkownika {user_id}")
        return
    
    print("\n--- Użytkownik ---")
    print(user.T)
    
    user_transactions = transactions_df[transactions_df['user_id'] == user_id]
    print(f"\nTransakcje: {len(user_transactions)}")
    print(f"Fraudy: {user_transactions['is_fraud'].sum()}")
    
    print("\nOstatnie 5 transakcji:")
    print(user_transactions.sort_values('timestamp', ascending=False).head())

def analyze_merchant(merchant_id):
    merchant = merchants_df[merchants_df['merchant_id'] == merchant_id]
    if merchant.empty:
        print(f"Nie znaleziono sprzedawcy {merchant_id}")
        return
    
    print("\n--- Sprzedawca ---")
    print(merchant.T)
    
    merchant_transactions = transactions_df[transactions_df['merchant_id'] == merchant_id]
    print(f"\nTransakcje: {len(merchant_transactions)}")
    print(f"Fraudy: {100 * merchant_transactions['is_fraud'].mean():.4f}%")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(merchant_transactions['amount'], bins=20, kde=True)
    plt.title(f'Kwoty transakcji - {merchant_id}')
    plt.xlabel('Kwota')
    plt.ylabel('Liczba')
    plt.tight_layout()
    plt.show()

def feature_importance():
    print("\n--- Ważność cech ---")
    
    features = [
        'amount', 'channel', 'device', 'is_international', 
        'session_length_seconds', 'is_first_time_merchant',
        'hour', 'day_of_week'
    ]
    
    X = pd.get_dummies(transactions_df[features], drop_first=True)
    y = transactions_df['is_fraud']
    
    corrs = []
    for col in X.columns:
        c = np.corrcoef(X[col], y)[0, 1]
        corrs.append((col, abs(c)))
    
    corrs.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 15 cech:")
    for i, (feat, corr) in enumerate(corrs[:15], 1):
        print(f"{i}. {feat}: {corr:.4f}")

print("\n" + "="*80)
print("Interaktywna analiza:")
print("- users_df: Dane użytkowników")
print("- merchants_df: Dane sprzedawców")
print("- transactions_df: Dane transakcji")
print("- plot_fraud_by_feature(feature): Wykres fraudów")
print("- analyze_user(user_id): Analiza użytkownika")
print("- analyze_merchant(merchant_id): Analiza sprzedawcy")
print("- feature_importance(): Ważność cech")
print("="*80)
print("\nPrzykłady:")
print("plot_fraud_by_feature('payment_method')")
print("analyze_user('U00001')")
print("analyze_merchant('M0001')")
print("feature_importance()")

embed() 