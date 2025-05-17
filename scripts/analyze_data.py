#!/usr/bin/env python
# -*- coding: utf-8 -*-

from explore_data import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Ustawienia wizualizacji
plt.style.use('ggplot')
sns.set_palette('viridis')

def parse_timestamp(df):
    """Konwertuje kolumnę timestamp na format datetime."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    return df

def fraud_analysis(transactions_df):
    """Przeprowadza podstawową analizę fraudów w danych."""
    print("\n--- Analiza Fraudów ---")
    
    # Podstawowe statystyki
    fraud_count = transactions_df['is_fraud'].sum()
    total_count = len(transactions_df)
    print(f"Fraudy: {fraud_count}/{total_count} ({100 * fraud_count / total_count:.4f}%)")
    
    # Fraud według kanału
    plt.figure(figsize=(10, 6))
    fraud_by_channel = transactions_df.groupby('channel')['is_fraud'].mean().sort_values(ascending=False)
    fraud_by_channel.plot(kind='bar', color='crimson')
    plt.title('Fraudy według kanału')
    plt.ylabel('Procent')
    plt.xlabel('Kanał')
    plt.tight_layout()
    plt.savefig('fraud_by_channel.png')
    
    # Fraud według metody płatności
    plt.figure(figsize=(10, 6))
    fraud_by_payment = transactions_df.groupby('payment_method')['is_fraud'].mean().sort_values(ascending=False)
    fraud_by_payment.plot(kind='bar', color='navy')
    plt.title('Fraudy według metody płatności')
    plt.ylabel('Procent')
    plt.xlabel('Metoda')
    plt.tight_layout()
    plt.savefig('fraud_by_payment.png')
    
    # Fraud według czasu dnia
    plt.figure(figsize=(12, 6))
    fraud_by_hour = transactions_df.groupby('hour')['is_fraud'].mean()
    fraud_by_hour.plot(kind='line', marker='o', linewidth=2, color='darkgreen')
    plt.title('Fraudy według godziny')
    plt.ylabel('Procent')
    plt.xlabel('Godzina')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fraud_by_hour.png')
    
    # Fraud według dnia tygodnia
    plt.figure(figsize=(10, 6))
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fraud_by_day = transactions_df.groupby('day_of_week')['is_fraud'].mean().reindex(order)
    fraud_by_day.plot(kind='bar', color='purple')
    plt.title('Fraudy według dnia tygodnia')
    plt.ylabel('Procent')
    plt.xlabel('Dzień')
    plt.tight_layout()
    plt.savefig('fraud_by_day.png')
    
    # Fraud według kwoty transakcji
    plt.figure(figsize=(10, 6))
    # Sprawdźmy zakres kwot transakcji
    min_amount = transactions_df['amount'].min()
    max_amount = transactions_df['amount'].max()
    print(f"Zakres kwot transakcji: {min_amount} - {max_amount}")
    
    # Zdefiniuj biny odpowiednio do zakresu danych
    bins = [0, 25, 50, 100, 200, 300, max_amount + 1]
    labels = ['0-25', '25-50', '50-100', '100-200', '200-300', '300+']
    
    print("Biny dla kwot transakcji:", bins)
    
    # Grupowanie transakcji według kwoty
    fraud_by_amount = transactions_df.groupby(pd.cut(transactions_df['amount'], bins=bins))['is_fraud'].mean()
    fraud_by_amount.index = labels  # Przypisz etykiety do indeksu
    
    # Wykres
    fraud_by_amount.plot(kind='bar', color='tomato')
    plt.title('Fraudy według kwoty')
    plt.ylabel('Procent')
    plt.xlabel('Kwota')
    plt.tight_layout()
    plt.savefig('fraud_by_amount.png')

def user_analysis(users_df, transactions_df):
    """Przeprowadza podstawową analizę użytkowników."""
    print("\n--- Analiza Użytkowników ---")
    
    # Podstawowe statystyki
    print(f"Liczba użytkowników: {len(users_df)}")
    
    # Rozkład wieku
    plt.figure(figsize=(10, 6))
    sns.histplot(users_df['age'], bins=20, kde=True)
    plt.title('Rozkład wieku')
    plt.xlabel('Wiek')
    plt.ylabel('Liczba')
    plt.tight_layout()
    plt.savefig('user_age_distribution.png')
    
    # Rozkład płci
    plt.figure(figsize=(8, 8))
    users_df['sex'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Rozkład płci')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('user_gender_distribution.png')
    
    # Średnia liczba transakcji według płci
    plt.figure(figsize=(8, 6))
    # Liczba transakcji na użytkownika
    tx_per_user = transactions_df.groupby('user_id').size().reset_index(name='tx_count')
    # Połączenie z danymi o użytkownikach
    tx_per_user = tx_per_user.merge(users_df[['user_id', 'sex']], on='user_id', how='left')
    # Wykres
    sns.boxplot(x='sex', y='tx_count', data=tx_per_user)
    plt.title('Transakcje według płci')
    plt.xlabel('Płeć')
    plt.ylabel('Liczba')
    plt.tight_layout()
    plt.savefig('tx_count_by_gender.png')
    
    # Ryzyko użytkownika vs fraudy
    plt.figure(figsize=(10, 6))
    # Identyfikacja użytkowników z fraudami
    users_with_fraud = transactions_df[transactions_df['is_fraud'] == 1]['user_id'].unique()
    users_df['had_fraud'] = users_df['user_id'].isin(users_with_fraud).astype(int)
    
    # Wykres pudełkowy risk_score według wystąpienia fraudu
    sns.boxplot(x='had_fraud', y='risk_score', data=users_df)
    plt.title('Risk Score vs Fraud')
    plt.xlabel('Fraud (0=Nie, 1=Tak)')
    plt.ylabel('Risk Score')
    plt.tight_layout()
    plt.savefig('risk_score_vs_fraud.png')

def merchant_analysis(merchants_df, transactions_df):
    """Przeprowadza podstawową analizę sprzedawców."""
    print("\n--- Analiza Sprzedawców ---")
    
    # Podstawowe statystyki
    print(f"Liczba sprzedawców: {len(merchants_df)}")
    
    # Rozkład kategorii
    plt.figure(figsize=(12, 6))
    merchants_df['category'].value_counts().plot(kind='bar')
    plt.title('Sprzedawcy według kategorii')
    plt.xlabel('Kategoria')
    plt.ylabel('Liczba')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('merchant_category_distribution.png')
    
    # Trust score vs fraud history
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='has_fraud_history', y='trust_score', data=merchants_df)
    plt.title('Trust Score vs Historia fraudów')
    plt.xlabel('Historia (0=Nie, 1=Tak)')
    plt.ylabel('Trust Score')
    plt.tight_layout()
    plt.savefig('trust_score_vs_fraud_history.png')
    
    # Średnia kwota transakcji według kategorii
    plt.figure(figsize=(12, 6))
    merged_data = transactions_df.merge(merchants_df[['merchant_id', 'category']], on='merchant_id', how='left')
    avg_amount_by_category = merged_data.groupby('category')['amount'].mean().sort_values(ascending=False)
    avg_amount_by_category.plot(kind='bar', color='teal')
    plt.title('Średnia kwota według kategorii')
    plt.xlabel('Kategoria')
    plt.ylabel('Kwota (EUR)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('avg_amount_by_category.png')
    
    # Fraud rate według kategorii
    plt.figure(figsize=(12, 6))
    fraud_by_category = merged_data.groupby('category')['is_fraud'].mean().sort_values(ascending=False)
    fraud_by_category.plot(kind='bar', color='darkred')
    plt.title('Fraudy według kategorii')
    plt.xlabel('Kategoria')
    plt.ylabel('Procent')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('fraud_by_category.png')

if __name__ == "__main__":
    # Wczytanie danych
    users_df, merchants_df, transactions_df = load_data()
    
    # Parsowanie dat
    transactions_df = parse_timestamp(transactions_df)
    
    # Przeprowadzenie analiz
    fraud_analysis(transactions_df)
    user_analysis(users_df, transactions_df)
    merchant_analysis(merchants_df, transactions_df)
    
    print("\nAnaliza zakończona. Wykresy zapisane.") 